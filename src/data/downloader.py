"""
Gene-disease dataset downloader.

Sources (all freely available, no registration required)
---------------------------------------------------------
1. NCBI ClinVar ``gene_condition_source_id``
   URL: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id
   Columns: GeneID | AssociatedGenes | RelatedGenes | ConceptID | DiseaseName | ...

2. NHGRI-EBI GWAS Catalog (all associations)
   URL: https://www.ebi.ac.uk/gwas/api/search/downloads/associations
   Covers GWAS-identified gene–trait associations from published studies.

3. Human Phenotype Ontology (HPO) ``genes_to_disease.txt``  [fallback]
   URL: https://purl.obolibrary.org/obo/hp/hpoa/genes_to_disease.txt
   Columns: ncbi_gene_id | gene_symbol | disease_id | disease_name
"""

import gzip
import io
import logging
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Public dataset URLs (no authentication required) ────────────────────────

NCBI_CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id"
)
GWAS_CATALOG_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/gwas-catalog-associations-full.zip"
)
HPO_GENES_URL = (
    "https://purl.obolibrary.org/obo/hp/hpoa/genes_to_disease.txt"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; GeneDiseaseRecommender/1.0; "
        "+https://github.com/ShivaniPimparkar111/biology-recommender)"
    )
}


# ── Low-level helpers ────────────────────────────────────────────────────────

class _TqdmUpTo(tqdm):
    """Provides ``update_to`` hook for urlretrieve progress."""

    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _http_get(url: str, dest: str) -> str:
    """Download *url* to *dest*, showing a progress bar."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    logger.info("GET %s → %s", url, dest)
    with _TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=Path(dest).name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)  # noqa: S310

    return dest


def _extract_gwas_zip(zip_path: str, dest_tsv: str) -> None:
    """Extract the first TSV from a GWAS Catalog ZIP download."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        tsv_members = [m for m in zf.namelist() if m.endswith(".tsv")]
        if not tsv_members:
            raise ValueError(f"No TSV found inside {zip_path}")
        member = sorted(tsv_members)[0]
        logger.info("Extracting %s → %s", member, dest_tsv)
        with zf.open(member) as src, open(dest_tsv, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _decompress_gz(gz_path: str) -> str:
    """Decompress a .gz file in-place; returns path without the .gz suffix."""
    out_path = gz_path[:-3]
    logger.info("Decompressing %s → %s", gz_path, out_path)
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return out_path


# ── Main public function ─────────────────────────────────────────────────────

def load_raw_data(raw_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download (if necessary) and parse all available gene–disease datasets,
    then merge them into a single DataFrame.

    Sources attempted (in order):
      1. NCBI ClinVar gene_condition_source_id
      2. NHGRI-EBI GWAS Catalog (full associations download)
      3. HPO genes_to_disease.txt (fallback if both above fail)

    Parameters
    ----------
    raw_dir : str
        Directory where downloaded files are cached.

    Returns
    -------
    pd.DataFrame
        Merged raw dataframe with columns ``gene`` and ``disease``.
    """
    os.makedirs(raw_dir, exist_ok=True)
    frames = []

    # ── 1. NCBI ClinVar ──────────────────────────────────────────────────────
    ncbi_path = os.path.join(raw_dir, "gene_condition_source_id.txt")
    try:
        if not os.path.exists(ncbi_path):
            _http_get(NCBI_CLINVAR_URL, ncbi_path)
        df_clinvar = _parse_ncbi_clinvar(ncbi_path)
        logger.info("ClinVar: %d raw rows", len(df_clinvar))
        frames.append(df_clinvar)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ClinVar failed (%s).", exc)

    # ── 2. GWAS Catalog (disabled) ───────────────────────────────────────────
    # GWAS Catalog adds 1M+ rows of GWAS traits (e.g. "Height", "BMI") which
    # are not clinical disease conditions and bloat the model to unusable size.
    # Re-enable by removing the `if False` guard when you want GWAS traits.
    if False:  # noqa: SIM210
        gwas_path = os.path.join(raw_dir, "gwas_catalog_associations.tsv")
        gwas_zip  = os.path.join(raw_dir, "gwas_catalog_associations.zip")
        try:
            if not os.path.exists(gwas_path):
                _http_get(GWAS_CATALOG_URL, gwas_zip)
                _extract_gwas_zip(gwas_zip, gwas_path)
            df_gwas = _parse_gwas_catalog(gwas_path)
            logger.info("GWAS Catalog: %d raw rows", len(df_gwas))
            frames.append(df_gwas)
        except Exception as exc:  # noqa: BLE001
            logger.warning("GWAS Catalog failed (%s). Skipping.", exc)

    # ── 3. HPO fallback (only if nothing else loaded) ────────────────────────
    if not frames:
        hpo_path = os.path.join(raw_dir, "genes_to_disease.txt")
        try:
            if not os.path.exists(hpo_path):
                _http_get(HPO_GENES_URL, hpo_path)
            df_hpo = _parse_hpo(hpo_path)
            logger.info("HPO fallback: %d raw rows", len(df_hpo))
            frames.append(df_hpo)
        except Exception as exc:  # noqa: BLE001
            logger.error("HPO fallback also failed (%s).", exc)
            raise RuntimeError("All data sources failed. Cannot continue.") from exc

    merged = pd.concat(frames, ignore_index=True)
    logger.info("Total merged raw rows: %d", len(merged))
    return merged


# ── Format-specific parsers ──────────────────────────────────────────────────

def _parse_ncbi_clinvar(path: str) -> pd.DataFrame:
    """
    Parse NCBI ClinVar ``gene_condition_source_id``.

    Actual columns (9, tab-separated, header commented with #):
        GeneID | AssociatedGenes | RelatedGenes | ConceptID | DiseaseName
        SourceName | SourceID | DiseaseMIM | LastUpdated

    Both AssociatedGenes (primary) and RelatedGenes (modifier/secondary)
    are included as separate gene–disease pairs.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        index_col=False,
        names=[
            "gene_id", "gene", "related_gene", "concept_id", "disease",
            "source_name", "source_id", "disease_mim", "last_updated",
        ],
        dtype=str,
        low_memory=False,
    )

    # Primary associations (AssociatedGenes → DiseaseName)
    primary = df[["gene", "disease"]].dropna(subset=["gene", "disease"])

    # Related gene associations (RelatedGenes → DiseaseName) where non-null
    related = (
        df[["related_gene", "disease"]]
        .dropna(subset=["related_gene", "disease"])
        .rename(columns={"related_gene": "gene"})
    )
    related = related[related["gene"].str.strip() != ""]

    return pd.concat([primary, related], ignore_index=True)


def _parse_gwas_catalog(path: str) -> pd.DataFrame:
    """
    Parse the NHGRI-EBI GWAS Catalog full associations TSV.

    Relevant columns:
        MAPPED_GENE   – gene(s) mapped to the SNP (e.g. "BRCA2" or "BRCA2, TP53")
        DISEASE/TRAIT – reported trait / disease name
    """
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]

    gene_col    = next((c for c in df.columns if "MAPPED_GENE" in c), None)
    disease_col = next((c for c in df.columns if "DISEASE" in c or "TRAIT" in c), None)

    if gene_col is None or disease_col is None:
        raise ValueError(f"Cannot find MAPPED_GENE / DISEASE columns in: {list(df.columns[:10])}")

    df = df[[gene_col, disease_col]].copy()
    df.columns = ["gene", "disease"]
    df = df.dropna()

    # GWAS mapped genes can be comma/space separated lists — explode them
    df["gene"] = df["gene"].str.split(r"[,\s]+")
    df = df.explode("gene")
    df["gene"] = df["gene"].str.strip()
    df = df[df["gene"].str.len() > 1]  # drop single-char noise

    return df[["gene", "disease"]].copy()


def _parse_hpo(path: str) -> pd.DataFrame:
    """
    Parse HPO ``genes_to_disease.txt``.

    Expected columns (tab-separated):
        ncbi_gene_id  gene_symbol  disease_id  disease_name
    Lines beginning with '#' are comments.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        comment="#",
        header=None,
        names=["gene_id", "gene", "disease_id", "disease"],
        dtype=str,
        low_memory=False,
    )
    return df[["gene", "disease"]].copy()
