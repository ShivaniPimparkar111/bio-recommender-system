"""
Data cleaning and standardisation pipeline.

Steps
-----
1. Rename columns to canonical names ``gene`` / ``disease``.
2. Strip whitespace and collapse internal spaces.
3. Uppercase gene symbols (HGNC convention).
4. Title-case disease names.
5. Drop null, empty-string, and placeholder values.
6. Drop exact duplicates.
7. Optionally filter out non-disease annotations (e.g. HPO phenotypic terms).
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Strings that indicate a missing or non-informative annotation
_JUNK_PATTERNS = re.compile(
    r"^(not provided|not specified|other|unknown|n/a|-|nan|none)$",
    re.IGNORECASE,
)

# HPO identifiers and Orphanet identifiers that are phenotypic (not disease names)
_HPO_PREFIX = re.compile(r"^HP:\d+", re.IGNORECASE)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise a raw gene–disease DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least two columns that map to ``gene`` and ``disease``.
        Acceptable input column names are detected automatically.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with exactly the columns [``gene``, ``disease``].
    """
    df = df.copy()

    # ── 1. Normalise column names ────────────────────────────────────────────
    df.columns = [c.strip().lower().lstrip("#") for c in df.columns]

    gene_col    = _detect_column(df, ["gene", "genesymbol", "gene_symbol", "approved_symbol"])
    disease_col = _detect_column(df, ["disease", "diseasename", "disease_name", "phenotype"])

    if gene_col is None or disease_col is None:
        raise ValueError(
            f"Cannot detect gene/disease columns in: {list(df.columns)}"
        )

    df = df.rename(columns={gene_col: "gene", disease_col: "disease"})
    df = df[["gene", "disease"]].copy()

    original_len = len(df)

    # ── 2. String normalisation ──────────────────────────────────────────────
    df["gene"]    = df["gene"].astype(str).str.strip().str.upper()
    df["gene"]    = df["gene"].str.replace(r"\s+", " ", regex=True)

    df["disease"] = df["disease"].astype(str).str.strip()
    df["disease"] = df["disease"].str.replace(r"\s+", " ", regex=True)
    df["disease"] = df["disease"].str.title()

    # ── 3. Remove junk / placeholder values ─────────────────────────────────
    for col in ("gene", "disease"):
        df = df[~df[col].str.match(_JUNK_PATTERNS)]
        df = df[df[col].str.len() > 0]

    # ── 4. Remove raw HPO phenotypic term IDs used as disease names ──────────
    df = df[~df["disease"].str.match(_HPO_PREFIX)]

    # ── 5. Remove duplicates ─────────────────────────────────────────────────
    df = df.drop_duplicates()

    # ── 6. Drop diseases with only one known gene ────────────────────────────
    # Single-gene diseases have no disease "profile" for similarity ranking and
    # make the graph 10x larger, causing RWR queries to time out.
    disease_gene_counts = df.groupby("disease")["gene"].nunique()
    multi_gene_diseases = disease_gene_counts[disease_gene_counts >= 2].index
    before = len(df)
    df = df[df["disease"].isin(multi_gene_diseases)]
    logger.info(
        "clean_data: dropped %d single-gene-disease rows (%d diseases removed)",
        before - len(df),
        disease_gene_counts[disease_gene_counts < 2].shape[0],
    )

    logger.info(
        "clean_data: %d → %d rows  (removed %d)",
        original_len, len(df), original_len - len(df),
    )
    return df.reset_index(drop=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name present in *df*, else None."""
    for name in candidates:
        if name in df.columns:
            return name
    return None
