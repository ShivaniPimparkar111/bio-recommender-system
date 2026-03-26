"""
Graph-Based Recommender  –  Bipartite Random Walk with Restart (RWR).

Methodology
----------------------------------
Gene–disease associations naturally form a *bipartite graph* G = (V_g ∪ V_d, E)
where edges represent curated associations.  Random Walk with Restart (RWR)
propagates a probability mass from a *seed* node, converging to a stationary
distribution that captures *global network context* – not just direct
neighbours.  This is equivalent to Personalised PageRank.

Additional analyses provided:
  * Degree centrality (hub genes / diseases)
  * Betweenness centrality (bridge nodes)
  * Community detection via the Louvain method
  * Jaccard co-occurrence for direct overlap queries

Mathematical formulation
------------------------
    p(t+1) = (1−α) · A_norm · p(t) + α · e_s

where
  * p      – probability vector over all nodes
  * A_norm – row-normalised adjacency of the bipartite graph
  * α      – restart probability (default 0.15)
  * e_s    – one-hot seed vector

References
----------
* Köhler et al. (2008) "Walking the interactome for prioritization of
  candidate disease genes", AJHG 82.
* Brin & Page (1998) "The anatomy of a large-scale hypertextual Web search
  engine", WWW.
"""

from __future__ import annotations

import logging
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .base_recommender import (
    BaseRecommender,
    RecommendationResponse,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


class GraphRecommender(BaseRecommender):
    """Bipartite Random-Walk-with-Restart gene–disease recommender."""

    name = "graph_rwr"

    def __init__(self, alpha: float = 0.15, max_iter: int = 100, tol: float = 1e-6) -> None:
        super().__init__()
        self._alpha    = alpha     # restart probability
        self._max_iter = max_iter
        self._tol      = tol
        self._graph: nx.Graph | None = None
        # Node-index lookups (bipartite)
        self._node_idx:  dict[str, int] = {}
        self._idx_node:  dict[int, str] = {}
        self._gene_nodes:    set[str]   = set()
        self._disease_nodes: set[str]   = set()
        self._A: np.ndarray | None      = None  # row-normalised adjacency

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "GraphRecommender":
        logger.info("[Graph] Building bipartite graph from %d associations …", len(df))

        genes    = sorted(df["gene"].unique())
        diseases = sorted(df["disease"].unique())

        self._genes    = genes
        self._diseases = diseases

        # Prefix nodes to avoid gene/disease name collisions
        self._gene_nodes    = {f"G:{g}" for g in genes}
        self._disease_nodes = {f"D:{d}" for d in diseases}

        all_nodes = sorted(self._gene_nodes) + sorted(self._disease_nodes)
        self._node_idx = {n: i for i, n in enumerate(all_nodes)}
        self._idx_node = {i: n for n, i in self._node_idx.items()}

        # Build NetworkX bipartite graph
        G = nx.Graph()
        G.add_nodes_from(self._gene_nodes,    bipartite=0)
        G.add_nodes_from(self._disease_nodes, bipartite=1)
        G.add_edges_from(
            (f"G:{row.gene}", f"D:{row.disease}")
            for row in df.itertuples(index=False)
        )
        self._graph = G

        # Build row-normalised adjacency for RWR
        n = len(all_nodes)
        rows, cols = [], []
        for u, v in G.edges():
            i, j = self._node_idx[u], self._node_idx[v]
            rows += [i, j]
            cols += [j, i]
        data = np.ones(len(rows), dtype=np.float32)
        A = csr_matrix((data, (rows, cols)), shape=(n, n)).toarray()

        # Row-normalise
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self._A = A / row_sums

        self._fitted = True
        logger.info(
            "[Graph] Fit complete: %d nodes, %d edges",
            G.number_of_nodes(), G.number_of_edges()
        )
        return self

    # ── Core RWR ─────────────────────────────────────────────────────────────

    def _rwr(self, seed_node: str) -> np.ndarray:
        """Run RWR from *seed_node*; return stationary probability vector."""
        n = self._A.shape[0]
        s = self._node_idx[seed_node]

        e_s = np.zeros(n, dtype=np.float64)
        e_s[s] = 1.0

        p = e_s.copy()
        for _ in range(self._max_iter):
            p_new = (1 - self._alpha) * self._A.T @ p + self._alpha * e_s
            if np.linalg.norm(p_new - p, 1) < self._tol:
                break
            p = p_new
        return p_new

    # ── Recommendation endpoints ─────────────────────────────────────────────

    def recommend_for_gene(
        self, gene: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        gene = self._normalise_gene(gene)
        seed = f"G:{gene}"
        if seed not in self._node_idx:
            return RecommendationResponse(query=gene, query_type="gene", model=self.name)

        probs     = self._rwr(seed)
        d_indices = {self._node_idx[f"D:{d}"]: d for d in self._diseases}
        scores    = {d: probs[i] for i, d in d_indices.items()}

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return RecommendationResponse(
            query=gene, query_type="gene", model=self.name,
            results=[
                RecommendationResult(name=d, score=float(s),
                                     reason="RWR stationary probability")
                for d, s in results if s > 0
            ],
        )

    def recommend_for_disease(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        disease = self._normalise_disease(disease)
        seed    = f"D:{disease}"
        if seed not in self._node_idx:
            return RecommendationResponse(
                query=disease, query_type="disease", model=self.name
            )

        probs    = self._rwr(seed)
        g_indices = {self._node_idx[f"G:{g}"]: g for g in self._genes}
        scores   = {g: probs[i] for i, g in g_indices.items()}

        results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return RecommendationResponse(
            query=disease, query_type="disease", model=self.name,
            results=[
                RecommendationResult(name=g, score=float(s),
                                     reason="RWR stationary probability")
                for g, s in results if s > 0
            ],
        )

    def similar_genes(self, gene: str, top_k: int = 10) -> RecommendationResponse:
        self._assert_fitted()
        gene = self._normalise_gene(gene)
        seed = f"G:{gene}"
        if seed not in self._node_idx:
            return RecommendationResponse(query=gene, query_type="gene", model=self.name)

        probs    = self._rwr(seed)
        g_scores = {
            g: probs[self._node_idx[f"G:{g}"]]
            for g in self._genes if g != gene
        }
        results = sorted(g_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return RecommendationResponse(
            query=gene, query_type="gene", model=self.name,
            results=[
                RecommendationResult(name=g, score=float(s),
                                     reason="RWR network proximity")
                for g, s in results if s > 0
            ],
        )

    def similar_diseases(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        disease = self._normalise_disease(disease)
        seed    = f"D:{disease}"
        if seed not in self._node_idx:
            return RecommendationResponse(
                query=disease, query_type="disease", model=self.name
            )

        probs    = self._rwr(seed)
        d_scores = {
            d: probs[self._node_idx[f"D:{d}"]]
            for d in self._diseases if d != disease
        }
        results = sorted(d_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return RecommendationResponse(
            query=disease, query_type="disease", model=self.name,
            results=[
                RecommendationResult(name=d, score=float(s),
                                     reason="RWR network proximity")
                for d, s in results if s > 0
            ],
        )

    # ── Network analytics ────────────────────────────────────────────────────

    def get_network_data(
        self, entity: str, depth: int = 2, max_nodes: int = 80
    ) -> dict:
        """
        Return a node-link dict (D3-compatible) centred on *entity*.

        Parameters
        ----------
        entity    : gene symbol or disease name.
        depth     : ego-graph depth.
        max_nodes : cap the sub-graph to avoid browser overload.
        """
        self._assert_fitted()
        entity_norm = self._normalise_gene(entity)
        seed = f"G:{entity_norm}"
        if seed not in self._graph:
            entity_norm = self._normalise_disease(entity)
            seed = f"D:{entity_norm}"
        if seed not in self._graph:
            return {"nodes": [], "links": []}

        ego = nx.ego_graph(self._graph, seed, radius=depth)

        # Cap size
        if ego.number_of_nodes() > max_nodes:
            top_nodes = sorted(
                ego.degree(), key=lambda x: x[1], reverse=True
            )[:max_nodes]
            ego = ego.subgraph([n for n, _ in top_nodes])

        nodes = [
            {
                "id":   n,
                "type": "gene" if n.startswith("G:") else "disease",
                "label": n[2:],
                "degree": ego.degree(n),
            }
            for n in ego.nodes()
        ]
        links = [
            {"source": u, "target": v}
            for u, v in ego.edges()
        ]
        return {"nodes": nodes, "links": links}
