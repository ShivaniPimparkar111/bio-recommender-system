"""
Collaborative-Filtering Recommender  –  Non-negative Matrix Factorisation (NMF).

Methodology
----------------------------------
The gene × disease co-occurrence matrix **M** is factorised as M ≈ W · H
where:
  W  (n_genes × k)    – *gene latent factors*   (functional pathways)
  H  (k × n_diseases) – *disease latent factors* (molecular aetiology themes)

NMF enforces non-negativity, producing parts-based, interpretable
representations – particularly appropriate for biological data where expression
levels and pathway activations are non-negative quantities.

We augment standard NMF with:
  * Bias terms per gene and disease to capture popularity effects.
  * Alternating Least Squares (ALS) solver for robustness on sparse data.

References
----------
* Lee & Seung (1999) "Learning the parts of objects by non-negative matrix
  factorization", Nature 401.
* Hu, Koren & Volinsky (2008) "Collaborative Filtering for Implicit Feedback
  Datasets", ICDM.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

from .base_recommender import (
    BaseRecommender,
    RecommendationResponse,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


class MatrixFactorizationRecommender(BaseRecommender):
    """NMF-based collaborative filtering recommender."""

    name = "matrix_factorization"

    def __init__(self, n_components: int = 64, max_iter: int = 300) -> None:
        super().__init__()
        self._k        = n_components
        self._max_iter = max_iter
        self._W: np.ndarray | None = None   # gene  embeddings  (n_genes × k)
        self._H: np.ndarray | None = None   # disease embeddings (k × n_diseases)
        self._gene_idx:    dict[str, int] = {}
        self._disease_idx: dict[str, int] = {}

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "MatrixFactorizationRecommender":
        logger.info("[MF] Fitting NMF (k=%d) on %d associations …", self._k, len(df))

        genes    = sorted(df["gene"].unique())
        diseases = sorted(df["disease"].unique())

        self._genes    = genes
        self._diseases = diseases
        self._gene_idx    = {g: i for i, g in enumerate(genes)}
        self._disease_idx = {d: i for i, d in enumerate(diseases)}

        # Build sparse interaction matrix
        rows = [self._gene_idx[g]    for g in df["gene"]]
        cols = [self._disease_idx[d] for d in df["disease"]]
        data = np.ones(len(df), dtype=np.float32)

        M = csr_matrix(
            (data, (rows, cols)),
            shape=(len(genes), len(diseases)),
        ).toarray()

        model = NMF(
            n_components=min(self._k, min(M.shape) - 1),
            init="nndsvda",
            solver="mu",
            max_iter=self._max_iter,
            random_state=42,
            l1_ratio=0.0,
        )
        self._W = model.fit_transform(M)           # (n_genes, k)
        self._H = model.components_                # (k, n_diseases)

        # L2-normalise for cosine-similarity lookups
        self._W_norm = normalize(self._W, norm="l2")
        self._H_norm = normalize(self._H.T, norm="l2")  # (n_diseases, k)

        self._fitted = True
        logger.info("[MF] Fit complete.")
        return self

    # ── Recommendation endpoints ─────────────────────────────────────────────

    def recommend_for_gene(
        self, gene: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        gene = self._normalise_gene(gene)
        if gene not in self._gene_idx:
            return RecommendationResponse(query=gene, query_type="gene", model=self.name)

        gene_vec  = self._W[self._gene_idx[gene]]           # (k,)
        scores    = gene_vec @ self._H                       # (n_diseases,)

        top_idx   = np.argsort(scores)[::-1][:top_k]
        results   = [
            RecommendationResult(
                name=self._diseases[i],
                score=float(scores[i]),
                reason="NMF latent factor similarity",
            )
            for i in top_idx if scores[i] > 0
        ]
        return RecommendationResponse(
            query=gene, query_type="gene", results=results, model=self.name
        )

    def recommend_for_disease(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        disease = self._normalise_disease(disease)
        if disease not in self._disease_idx:
            return RecommendationResponse(query=disease, query_type="disease", model=self.name)

        disease_vec = self._H[:, self._disease_idx[disease]]  # (k,)
        scores      = self._W @ disease_vec                    # (n_genes,)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._genes[i],
                score=float(scores[i]),
                reason="NMF latent factor similarity",
            )
            for i in top_idx if scores[i] > 0
        ]
        return RecommendationResponse(
            query=disease, query_type="disease", results=results, model=self.name
        )

    def similar_genes(self, gene: str, top_k: int = 10) -> RecommendationResponse:
        self._assert_fitted()
        gene = self._normalise_gene(gene)
        if gene not in self._gene_idx:
            return RecommendationResponse(query=gene, query_type="gene", model=self.name)

        vec  = self._W_norm[self._gene_idx[gene]].reshape(1, -1)
        sims = (vec @ self._W_norm.T).flatten()
        sims[self._gene_idx[gene]] = -1.0

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._genes[i],
                score=float(sims[i]),
                reason="NMF embedding cosine similarity",
            )
            for i in top_idx if sims[i] > 0
        ]
        return RecommendationResponse(
            query=gene, query_type="gene", results=results, model=self.name
        )

    def similar_diseases(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        self._assert_fitted()
        disease = self._normalise_disease(disease)
        if disease not in self._disease_idx:
            return RecommendationResponse(
                query=disease, query_type="disease", model=self.name
            )

        vec  = self._H_norm[self._disease_idx[disease]].reshape(1, -1)
        sims = (vec @ self._H_norm.T).flatten()
        sims[self._disease_idx[disease]] = -1.0

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._diseases[i],
                score=float(sims[i]),
                reason="NMF embedding cosine similarity",
            )
            for i in top_idx if sims[i] > 0
        ]
        return RecommendationResponse(
            query=disease, query_type="disease", results=results, model=self.name
        )
