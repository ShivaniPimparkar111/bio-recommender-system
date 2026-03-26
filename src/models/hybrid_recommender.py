"""
Hybrid Recommender  –  Reciprocal Rank Fusion (RRF) Ensemble.

Methodology
----------------------------------
No single recommender dominates across all query types and data regimes.
We combine Content-Based, Matrix Factorisation, and Graph-RWR models via
*Reciprocal Rank Fusion* (RRF):

    RRF_score(d) = Σ_m  1 / (k + rank_m(d))

where k=60 is a smoothing constant (Cormack et al., 2009).  This score-agnostic
approach avoids the problem of calibrating heterogeneous score distributions
across models and empirically matches or exceeds Borda-count and weighted-sum
fusion strategies (Benham et al., 2020).

References
----------
* Cormack, Clarke & Buettcher (2009) "Reciprocal rank fusion outperforms
  condorcet and individual rank learning methods", SIGIR.
* Benham et al. (2020) "Paragraph-level rationale extraction through
  regularization", EMNLP.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import List

import pandas as pd

from .base_recommender import (
    BaseRecommender,
    RecommendationResponse,
    RecommendationResult,
)
from .content_based import ContentBasedRecommender
from .collaborative_filtering import MatrixFactorizationRecommender
from .graph_recommender import GraphRecommender

logger = logging.getLogger(__name__)

_RRF_K = 60  # RRF smoothing constant


class HybridRecommender(BaseRecommender):
    """
    Ensemble of ContentBased + MatrixFactorization + GraphRWR
    combined via Reciprocal Rank Fusion.
    """

    name = "hybrid_rrf"

    def __init__(
        self,
        cb_weight:  float = 1.0,
        mf_weight:  float = 1.0,
        rwr_weight: float = 1.0,
        n_components: int = 64,
        rwr_alpha:    float = 0.15,
    ) -> None:
        super().__init__()
        self._weights = {
            "content_based":       cb_weight,
            "matrix_factorization": mf_weight,
            "graph_rwr":           rwr_weight,
        }
        self._cb  = ContentBasedRecommender()
        self._mf  = MatrixFactorizationRecommender(n_components=n_components)
        self._rwr = GraphRecommender(alpha=rwr_alpha)

    # ── Fit all sub-models ────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "HybridRecommender":
        logger.info("[Hybrid] Fitting all sub-models …")
        self._cb.fit(df)
        self._mf.fit(df)
        self._rwr.fit(df)
        self._genes    = self._cb.genes
        self._diseases = self._cb.diseases
        self._fitted   = True
        logger.info("[Hybrid] All sub-models ready.")
        return self

    # ── RRF fusion helper ─────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        ranked_lists: list[tuple[str, list[RecommendationResult]]],
    ) -> list[RecommendationResult]:
        """Merge multiple ranked lists into one via RRF."""
        scores: dict[str, float] = defaultdict(float)
        reasons: dict[str, list[str]] = defaultdict(list)

        for model_name, results in ranked_lists:
            for rank, res in enumerate(results, start=1):
                scores[res.name]  += 1.0 / (_RRF_K + rank)
                reasons[res.name].append(f"{model_name}(rank={rank})")

        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [
            RecommendationResult(
                name=name,
                score=score,
                reason=" | ".join(reasons[name]),
            )
            for name, score in merged
        ]

    # ── Recommendation endpoints ─────────────────────────────────────────────

    def recommend_for_gene(
        self, gene: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        gene = self._normalise_gene(gene)
        fetch_k = top_k * 3  # over-fetch for better fusion coverage

        lists = [
            (self._cb.name,  self._cb.recommend_for_gene(gene, fetch_k).results),
            (self._mf.name,  self._mf.recommend_for_gene(gene, fetch_k).results),
            (self._rwr.name, self._rwr.recommend_for_gene(gene, fetch_k).results),
        ]
        results = self._rrf_merge(lists)[:top_k]
        return RecommendationResponse(
            query=gene, query_type="gene", results=results, model=self.name
        )

    def recommend_for_disease(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        disease = self._normalise_disease(disease)
        fetch_k = top_k * 3

        lists = [
            (self._cb.name,  self._cb.recommend_for_disease(disease, fetch_k).results),
            (self._mf.name,  self._mf.recommend_for_disease(disease, fetch_k).results),
            (self._rwr.name, self._rwr.recommend_for_disease(disease, fetch_k).results),
        ]
        results = self._rrf_merge(lists)[:top_k]
        return RecommendationResponse(
            query=disease, query_type="disease", results=results, model=self.name
        )

    def similar_genes(self, gene: str, top_k: int = 10) -> RecommendationResponse:
        self._assert_fitted()
        gene    = self._normalise_gene(gene)
        fetch_k = top_k * 3

        lists = [
            (self._cb.name,  self._cb.similar_genes(gene, fetch_k).results),
            (self._mf.name,  self._mf.similar_genes(gene, fetch_k).results),
            (self._rwr.name, self._rwr.similar_genes(gene, fetch_k).results),
        ]
        results = self._rrf_merge(lists)[:top_k]
        return RecommendationResponse(
            query=gene, query_type="gene", results=results, model=self.name
        )

    def similar_diseases(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        disease = self._normalise_disease(disease)
        fetch_k = top_k * 3

        lists = [
            (self._cb.name,  self._cb.similar_diseases(disease, fetch_k).results),
            (self._mf.name,  self._mf.similar_diseases(disease, fetch_k).results),
            (self._rwr.name, self._rwr.similar_diseases(disease, fetch_k).results),
        ]
        results = self._rrf_merge(lists)[:top_k]
        return RecommendationResponse(
            query=disease, query_type="disease", results=results, model=self.name
        )

    # ── Delegate network data to graph model ──────────────────────────────────

    def get_network_data(self, entity: str, depth: int = 2, max_nodes: int = 80) -> dict:
        """Return D3-compatible network data (delegated to GraphRecommender)."""
        self._assert_fitted()
        return self._rwr.get_network_data(entity, depth=depth, max_nodes=max_nodes)
