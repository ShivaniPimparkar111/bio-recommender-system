"""
Vignette: How to use the Gene–Disease Recommender System
=========================================================
This script walks through the complete workflow end-to-end using real data.

Run it from the project root after the data pipeline has been executed:

    python main.py --pipeline     # one-time download + clean
    python vignette.py            # this script

Each section prints its output so you can follow along interactively.
"""

import sys
import textwrap
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Load the cleaned dataset
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 1 — Load the cleaned gene–disease dataset")
print("=" * 65)

CSV_PATH = "data/gene_disease.csv"

try:
    df = pd.read_csv(CSV_PATH, dtype=str)
except FileNotFoundError:
    sys.exit(
        f"\n[ERROR] '{CSV_PATH}' not found.\n"
        "Run  `python main.py --pipeline`  first to download and clean the data.\n"
    )

print(f"\nDataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns       : {list(df.columns)}")
print(f"Unique genes  : {df['gene'].nunique():,}")
print(f"Unique diseases: {df['disease'].nunique():,}")
print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Explore a specific gene
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 2 — Explore a specific gene (BRCA2)")
print("=" * 65)

QUERY_GENE = "BRCA2"

brca2_diseases = (
    df[df["gene"] == QUERY_GENE]["disease"]
    .sort_values()
    .reset_index(drop=True)
)

print(f"\nGene '{QUERY_GENE}' is associated with {len(brca2_diseases)} diseases:")
for i, d in enumerate(brca2_diseases, 1):
    print(f"  {i:2}. {d}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Fit the Hybrid Recommender
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 3 — Fit the Hybrid Recommender (TF-IDF + NMF + Graph RWR)")
print("=" * 65)

from src.models import HybridRecommender

print("\nFitting all three sub-models … (may take ~5 seconds)")
model = HybridRecommender(n_components=min(64, df["gene"].nunique() - 1))
model.fit(df)
print("Done.")

print(f"\nModel covers {len(model.genes):,} genes and {len(model.diseases):,} diseases.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Disease recommendations for a gene
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 4 — Top-10 disease recommendations for BRCA2")
print("=" * 65)

response = model.recommend_for_gene(QUERY_GENE, top_k=10)

print(f"\nQuery : gene = '{response.query}'")
print(f"Model : {response.model}\n")
print(f"{'Rank':<5} {'Disease':<50} {'Score':>8}  Reason")
print("-" * 95)
for r in response.results:
    reason_short = r.reason[:40] + "…" if len(r.reason) > 40 else r.reason
    print(f"  {response.results.index(r)+1:<3} {r.name:<50} {r.score:>8.4f}  {reason_short}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Gene recommendations for a disease
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 5 — Top-10 gene recommendations for 'Breast Neoplasm'")
print("=" * 65)

QUERY_DISEASE = "Breast Neoplasm"
response2 = model.recommend_for_disease(QUERY_DISEASE, top_k=10)

print(f"\nQuery : disease = '{response2.query}'")
print(f"Model : {response2.model}\n")
print(f"{'Rank':<5} {'Gene':<20} {'Score':>8}  Reason")
print("-" * 70)
for r in response2.results:
    reason_short = r.reason[:38] + "…" if len(r.reason) > 38 else r.reason
    print(f"  {response2.results.index(r)+1:<3} {r.name:<20} {r.score:>8.4f}  {reason_short}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – Find genes similar to BRCA2
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 6 — Genes most similar to BRCA2 (shared disease profile)")
print("=" * 65)

sim_response = model.similar_genes(QUERY_GENE, top_k=8)

print(f"\nGenes most functionally similar to '{QUERY_GENE}':\n")
print(f"{'Rank':<5} {'Gene':<20} {'Score':>8}")
print("-" * 38)
for i, r in enumerate(sim_response.results, 1):
    print(f"  {i:<3} {r.name:<20} {r.score:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – Compare individual models
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 7 — Compare individual model rankings for BRCA2")
print("=" * 65)

cb_resp  = model._cb.recommend_for_gene(QUERY_GENE,  top_k=5)
mf_resp  = model._mf.recommend_for_gene(QUERY_GENE,  top_k=5)
rwr_resp = model._rwr.recommend_for_gene(QUERY_GENE, top_k=5)

print(f"\n{'TF-IDF (content-based)':<35} {'NMF (matrix factor.)':<35} {'Graph RWR':<35}")
print("-" * 105)
for i in range(5):
    cb  = cb_resp.results[i].name[:32]  if i < len(cb_resp.results)  else "—"
    mf  = mf_resp.results[i].name[:32]  if i < len(mf_resp.results)  else "—"
    rwr = rwr_resp.results[i].name[:32] if i < len(rwr_resp.results) else "—"
    print(f"  {cb:<33}  {mf:<33}  {rwr:<33}")

print(f"\n→ Hybrid RRF blends all three lists into one consensus ranking.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 – Similar diseases
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 8 — Diseases most similar to 'Cardiomyopathy'")
print("=" * 65)

sim_d = model.similar_diseases("Cardiomyopathy", top_k=6)
print(f"\nDiseases with the most similar gene profile to 'Cardiomyopathy':\n")
print(f"{'Rank':<5} {'Disease':<55} {'Score':>8}")
print("-" * 72)
for i, r in enumerate(sim_d.results, 1):
    print(f"  {i:<3} {r.name:<55} {r.score:>8.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 – Network sub-graph
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 9 — Network neighbourhood of BRCA2 (depth=1)")
print("=" * 65)

network = model.get_network_data("BRCA2", depth=1, max_nodes=30)
genes_in_net    = [n for n in network["nodes"] if n["type"] == "gene"]
diseases_in_net = [n for n in network["nodes"] if n["type"] == "disease"]

print(f"\nNodes  : {len(network['nodes'])}  ({len(genes_in_net)} genes, {len(diseases_in_net)} diseases)")
print(f"Edges  : {len(network['links'])}")
print("\nGenes in sub-graph:")
print("  " + ", ".join(n["label"] for n in genes_in_net[:12]) +
      (" …" if len(genes_in_net) > 12 else ""))
print("\nDiseases in sub-graph:")
for n in diseases_in_net[:8]:
    print(f"  • {n['label']}")
if len(diseases_in_net) > 8:
    print(f"  … and {len(diseases_in_net)-8} more")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 – Quick API reference
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  SECTION 10 — REST API quick reference")
print("=" * 65)

print(textwrap.dedent("""
  Start the server:
    python main.py --serve          # http://localhost:8000
    Interactive docs → http://localhost:8000/docs

  Example curl commands:

  # List first 5 genes
  curl "http://localhost:8000/api/v1/genes?page_size=5"

  # Disease recommendations for BRCA2
  curl "http://localhost:8000/api/v1/genes/BRCA2/recommend?top_k=5"

  # Gene recommendations for Breast Neoplasm
  curl "http://localhost:8000/api/v1/diseases/Breast%20Neoplasm/recommend?top_k=5"

  # Genes similar to TP53
  curl "http://localhost:8000/api/v1/genes/TP53/similar?top_k=5"

  # Network graph for KRAS (depth 2)
  curl "http://localhost:8000/api/v1/genes/KRAS/network?depth=2&max_nodes=40"

  # Dataset statistics
  curl "http://localhost:8000/api/v1/stats"

  # Choose a specific model (hybrid_rrf | content_based | matrix_factorization | graph_rwr)
  curl "http://localhost:8000/api/v1/genes/EGFR/recommend?model_name=graph_rwr&top_k=5"
"""))

print("=" * 65)
print("  Vignette complete. All systems operational.")
print("=" * 65 + "\n")
