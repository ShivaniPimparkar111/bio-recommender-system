"""
Content-Based Recommender  –  TF-IDF + Cosine Similarity.

Methodology
----------------------------------
Each gene is represented as a *document* whose *words* are the diseases it is
associated with (and vice-versa for diseases).  TF-IDF weighting de-emphasises
diseases shared by many genes (analogous to stop-words in NLP) and amplifies
rare, specific associations – precisely the signal a clinician or researcher
cares about.  Cosine similarity in the resulting embedding space captures
*functional overlap* between entities.

References
----------
* Salton & McGill (1983) Introduction to Modern Information Retrieval.
* Cheng et al. (2014) "A bioinformatics approach to identify gene–disease
  associations via gene ontology", BMC Genomics.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .base_recommender import (
    BaseRecommender,
    RecommendationResponse,
    RecommendationResult,
)

logger = logging.getLogger(__name__)


class ContentBasedRecommender(BaseRecommender):
    """TF-IDF content-based gene–disease recommender."""

    name = "content_based"

    def __init__(self, min_df: int = 2) -> None:
        super().__init__()
        self._min_df = min_df          # discard entities with < min_df co-occurrences
        # Gene-space matrices
        self._gene_matrix: np.ndarray | None = None
        # Disease-space matrices
        self._disease_matrix: np.ndarray | None = None
        # Lookup dictionaries
        self._gene_idx: dict[str, int]    = {}
        self._disease_idx: dict[str, int] = {}

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ContentBasedRecommender":
        """
        Build TF-IDF embedding matrices for genes and diseases.

        We construct *two* co-occurrence matrices:
          - genes    × diseases  (rows = genes,    cols = diseases)
          - diseases × genes     (rows = diseases, cols = genes)

        TF-IDF is applied *row-wise* in each matrix so that each entity's
        profile is weighted by how informative its associations are.
        """
        logger.info("[ContentBased] Fitting on %d associations …", len(df))

        genes    = sorted(df["gene"].unique())
        diseases = sorted(df["disease"].unique())

        self._genes    = genes
        self._diseases = diseases

        self._gene_idx    = {g: i for i, g in enumerate(genes)}
        self._disease_idx = {d: i for i, d in enumerate(diseases)}

        # ── Build raw count matrices ─────────────────────────────────────────
        gene_mat    = np.zeros((len(genes),    len(diseases)), dtype=np.float32)
        disease_mat = np.zeros((len(diseases), len(genes)),    dtype=np.float32)

        for _, row in df.iterrows():
            gi = self._gene_idx[row["gene"]]
            di = self._disease_idx[row["disease"]]
            gene_mat[gi, di]    += 1.0
            disease_mat[di, gi] += 1.0

        # ── Apply TF-IDF weighting ───────────────────────────────────────────
        self._gene_matrix    = _tfidf(gene_mat)
        self._disease_matrix = _tfidf(disease_mat)

        self._fitted = True
        logger.info("[ContentBased] Fit complete: %d genes × %d diseases",
                    len(genes), len(diseases))
        return self

    # ── Recommendation endpoints ─────────────────────────────────────────────

    def recommend_for_gene(
        self, gene: str, top_k: int = 10
    ) -> RecommendationResponse:
        """
        Return top-k diseases for *gene*.

        Strategy: find genes with similar disease profiles (cosine sim in
        disease-feature space), then aggregate their disease association weights.
        This surfaces diseases linked to functionally similar genes – exactly
        the transitive inference a clinical researcher wants.
        """
        self._assert_fitted()
        gene = self._normalise_gene(gene)

        if gene not in self._gene_idx:
            return RecommendationResponse(query=gene, query_type="gene", model=self.name)

        # gene_matrix is (n_genes, n_diseases) – compare rows in disease-space
        gene_vec = self._gene_matrix[self._gene_idx[gene]].reshape(1, -1)   # (1, n_diseases)
        gene_sims = cosine_similarity(gene_vec, self._gene_matrix)[0]        # (n_genes,)
        gene_sims[self._gene_idx[gene]] = 0.0   # exclude self from aggregation

        # Weighted sum across all genes → disease relevance scores
        scores = gene_sims @ self._gene_matrix   # (n_diseases,)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._diseases[i],
                score=float(scores[i]),
                reason="TF-IDF similar-gene aggregation",
            )
            for i in top_idx if scores[i] > 0
        ]
        return RecommendationResponse(
            query=gene, query_type="gene", results=results, model=self.name
        )

    def recommend_for_disease(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        """
        Return top-k genes for *disease*.

        Strategy: find diseases with similar gene profiles (cosine sim in
        gene-feature space), then aggregate their gene association weights.
        """
        self._assert_fitted()
        disease = self._normalise_disease(disease)

        if disease not in self._disease_idx:
            return RecommendationResponse(query=disease, query_type="disease", model=self.name)

        # disease_matrix is (n_diseases, n_genes) – compare rows in gene-space
        disease_vec = self._disease_matrix[self._disease_idx[disease]].reshape(1, -1)  # (1, n_genes)
        disease_sims = cosine_similarity(disease_vec, self._disease_matrix)[0]          # (n_diseases,)
        disease_sims[self._disease_idx[disease]] = 0.0

        # Weighted sum across all diseases → gene relevance scores
        scores = disease_sims @ self._disease_matrix   # (n_genes,)

        top_idx = np.argsort(scores)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._genes[i],
                score=float(scores[i]),
                reason="TF-IDF similar-disease aggregation",
            )
            for i in top_idx if scores[i] > 0
        ]
        return RecommendationResponse(
            query=disease, query_type="disease", results=results, model=self.name
        )

    def similar_genes(self, gene: str, top_k: int = 10) -> RecommendationResponse:
        """Return genes whose disease profile is most similar to *gene*."""
        self._assert_fitted()
        gene = self._normalise_gene(gene)

        if gene not in self._gene_idx:
            return RecommendationResponse(query=gene, query_type="gene", model=self.name)

        vec = self._gene_matrix[self._gene_idx[gene]].reshape(1, -1)
        sims = cosine_similarity(vec, self._gene_matrix)[0]
        sims[self._gene_idx[gene]] = -1.0  # exclude self

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._genes[i],
                score=float(sims[i]),
                reason="Shared disease profile (TF-IDF cosine)",
            )
            for i in top_idx if sims[i] > 0
        ]
        return RecommendationResponse(
            query=gene, query_type="gene", results=results, model=self.name
        )

    def similar_diseases(
        self, disease: str, top_k: int = 10
    ) -> RecommendationResponse:
        """Return diseases whose gene profile is most similar to *disease*."""
        self._assert_fitted()
        disease = self._normalise_disease(disease)

        if disease not in self._disease_idx:
            return RecommendationResponse(
                query=disease, query_type="disease", model=self.name
            )

        vec = self._disease_matrix[self._disease_idx[disease]].reshape(1, -1)
        sims = cosine_similarity(vec, self._disease_matrix)[0]
        sims[self._disease_idx[disease]] = -1.0  # exclude self

        top_idx = np.argsort(sims)[::-1][:top_k]
        results = [
            RecommendationResult(
                name=self._diseases[i],
                score=float(sims[i]),
                reason="Shared gene profile (TF-IDF cosine)",
            )
            for i in top_idx if sims[i] > 0
        ]
        return RecommendationResponse(
            query=disease, query_type="disease", results=results, model=self.name
        )


# ── TF-IDF helper ────────────────────────────────────────────────────────────

def _tfidf(count_matrix: np.ndarray) -> np.ndarray:
    """
    Apply TF-IDF to *count_matrix* (rows = documents, cols = terms).

    TF  = raw count  (already 0/1 for binary data but generalises)
    IDF = log(N / df) with +1 smoothing to avoid zero division.
    Returns L2-normalised rows for cosine-friendly arithmetic.
    """
    n_docs = count_matrix.shape[0]
    df     = np.sum(count_matrix > 0, axis=0).astype(np.float32) + 1.0
    idf    = np.log(n_docs / df) + 1.0
    tfidf  = count_matrix * idf[np.newaxis, :]
    return normalize(tfidf, norm="l2", axis=1)
