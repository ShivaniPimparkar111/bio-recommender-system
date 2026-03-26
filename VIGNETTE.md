# Vignette: How to use the Gene–Disease Recommender

A step-by-step walkthrough of every major feature, shown with real examples
from the NCBI ClinVar dataset.

---

## Prerequisites

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download & clean the dataset (one-time, ~30 seconds)
python main.py --pipeline
```

You should see output like:

```
Step 1/3  Downloading raw data …
Step 2/3  Cleaning data …
Step 3/3  Saving to data/gene_disease.csv …
Dataset ready: {'total_associations': 2875, 'unique_genes': 1747, 'unique_diseases': 611, ...}
```

---

## Run the full vignette script

```bash
python vignette.py
```

This runs all 10 sections below automatically and prints their output.

---

## Section 1 — Load the dataset

```python
import pandas as pd
df = pd.read_csv("data/gene_disease.csv", dtype=str)

print(df.shape)          # (2875, 2)
print(df.head())
#      gene                              disease
# 0   BRCA2                        Fanconi Anemia
# 1   BRCA2                        Breast Neoplasm
# 2   BRCA2       Neoplasm Of The Genitourinary Tract
# ...
```

The cleaned CSV has exactly two columns: **`gene`** (HGNC symbol, uppercase)
and **`disease`** (title-case free text from NCBI ClinVar).

---

## Section 2 — Explore a gene

```python
brca2 = df[df["gene"] == "BRCA2"]["disease"].sort_values()
print(brca2.to_list())
# ['Breast Neoplasm', 'Fanconi Anemia', 'Hereditary Wilms Tumor',
#  'Malignant Tumor Of Breast', 'Neoplasm Of Breast', ...]
```

---

## Section 3 — Fit the Hybrid Recommender

```python
from src.models import HybridRecommender

model = HybridRecommender()
model.fit(df)
# [ContentBased] Fit complete: 1747 genes × 611 diseases
# [MF] Fit complete.
# [Graph] Fit complete: 2358 nodes, 2875 edges
# [Hybrid] All sub-models ready.
```

Behind the scenes this fits three algorithms simultaneously:

| Sub-model | What it learns |
|-----------|---------------|
| `ContentBasedRecommender` | TF-IDF vectors for each gene and disease |
| `MatrixFactorizationRecommender` | 64 latent factors via NMF |
| `GraphRecommender` | Bipartite graph for Random Walk with Restart |

---

## Section 4 — Disease recommendations for a gene

> **"Given gene BRCA2, which diseases is it most likely associated with?"**

```python
resp = model.recommend_for_gene("BRCA2", top_k=10)

for i, r in enumerate(resp.results, 1):
    print(f"  {i:2}. {r.name:<50}  score={r.score:.4f}")
```

**Example output:**

```
   1. Fanconi Anemia                                    score=0.0481
   2. Breast Neoplasm                                   score=0.0479
   3. Neoplasm Of The Genitourinary Tract               score=0.0474
   4. Malignant Tumor Of Breast                         score=0.0472
   5. Hereditary Wilms Tumor                            score=0.0459
   6. Ovarian Neoplasm                                  score=0.0452
   7. Hereditary Breast And Ovarian Cancer Syndrome     score=0.0448
   8. Malignant Neoplasm Of Ovary                       score=0.0441
   9. Pancreatic Cancer                                 score=0.0437
  10. Prostate Cancer                                   score=0.0431
```

Each `score` is a Reciprocal Rank Fusion score — a model-agnostic consensus
of all three underlying algorithms. The `reason` field shows each model's rank
contribution, e.g. `content_based(rank=2) | matrix_factorization(rank=1) | graph_rwr(rank=3)`.

---

## Section 5 — Gene recommendations for a disease

> **"Given disease 'Breast Neoplasm', which genes are most likely involved?"**

```python
resp = model.recommend_for_disease("Breast Neoplasm", top_k=10)

for i, r in enumerate(resp.results, 1):
    print(f"  {i:2}. {r.name:<20}  score={r.score:.4f}")
```

**Example output:**

```
   1. STK11                score=0.0479
   2. PTEN                 score=0.0477
   3. CHEK2                score=0.0476
   4. RAD51C               score=0.0469
   5. PALB2                score=0.0461
   6. BRCA1                score=0.0458
   7. BRCA2                score=0.0452
   8. CDH1                 score=0.0441
   9. ATM                  score=0.0435
  10. BARD1                score=0.0429
```

---

## Section 6 — Similar genes

> **"Which genes share a disease profile most similar to BRCA2?"**

```python
resp = model.similar_genes("BRCA2", top_k=8)

for i, r in enumerate(resp.results, 1):
    print(f"  {i}. {r.name}  (score={r.score:.4f})")
```

**Example output:**

```
  1. BRCA1   (score=0.0479)
  2. PALB2   (score=0.0461)
  3. RAD51C  (score=0.0447)
  4. CHEK2   (score=0.0438)
  5. ATM     (score=0.0431)
  6. BARD1   (score=0.0424)
  7. NBN     (score=0.0417)
  8. RAD51D  (score=0.0409)
```

These are all known BRCA-pathway / homologous-recombination genes — exactly what
you would expect from a biologically meaningful recommender.

---

## Section 7 — Compare individual models

You can access each sub-model directly to compare their raw rankings:

```python
cb  = model._cb.recommend_for_gene("BRCA2", top_k=5)
mf  = model._mf.recommend_for_gene("BRCA2", top_k=5)
rwr = model._rwr.recommend_for_gene("BRCA2", top_k=5)
```

| Rank | TF-IDF | NMF | Graph RWR |
|------|--------|-----|-----------|
| 1 | Breast Neoplasm | Fanconi Anemia | Breast Neoplasm |
| 2 | Malignant Tumor Of Breast | Neoplasm Of Genitourinary Tract | Malignant Tumor Of Breast |
| 3 | Fanconi Anemia | Breast Neoplasm | Fanconi Anemia |
| 4 | Neoplasm Of Genitourinary Tract | Hereditary Wilms Tumor | Neoplasm Of Genitourinary Tract |
| 5 | Hereditary Wilms Tumor | Pancreatic Cancer | Hereditary Wilms Tumor |

The **Hybrid RRF** aggregates these via `score = Σ 1/(60 + rank)` per model,
producing a single consensus list that is more robust than any individual model.

---

## Section 8 — Similar diseases

> **"Which diseases have the most similar genetic architecture to Cardiomyopathy?"**

```python
resp = model.similar_diseases("Cardiomyopathy", top_k=6)
for i, r in enumerate(resp.results, 1):
    print(f"  {i}. {r.name}  ({r.score:.4f})")
```

**Example output:**

```
  1. Primary Familial Dilated Cardiomyopathy      (0.0487)
  2. Primary Dilated Cardiomyopathy               (0.0474)
  3. Primary Familial Hypertrophic Cardiomyopathy (0.0457)
  4. Arrhythmogenic Right Ventricular Dysplasia   (0.0441)
  5. Hypertrophic Cardiomyopathy                  (0.0432)
  6. Left Ventricular Noncompaction               (0.0419)
```

---

## Section 9 — Network sub-graph

The Graph model exposes the underlying bipartite network. This is what powers
the D3 visualisation in the web UI.

```python
net = model.get_network_data("BRCA2", depth=1, max_nodes=30)

print(f"Nodes: {len(net['nodes'])}  Edges: {len(net['links'])}")
# Nodes: 21  Edges: 20

genes    = [n["label"] for n in net["nodes"] if n["type"] == "gene"]
diseases = [n["label"] for n in net["nodes"] if n["type"] == "disease"]

print("Genes in subgraph :", genes)
# ['BRCA2']

print("Diseases in subgraph:", diseases[:5])
# ['Breast Neoplasm', 'Fanconi Anemia', 'Hereditary Wilms Tumor', ...]
```

At `depth=2` the sub-graph expands to include genes that share a disease with BRCA2,
giving you a multi-hop neighbourhood view.

---

## Section 10 — REST API

Start the server then query it with curl or any HTTP client:

```bash
python main.py --serve
# Uvicorn running on http://0.0.0.0:8000
```

### Disease recommendations for a gene
```bash
curl "http://localhost:8000/api/v1/genes/BRCA2/recommend?top_k=5"
```
```json
{
  "query": "BRCA2",
  "model": "hybrid_rrf",
  "results": [
    {"name": "Fanconi Anemia",   "score": 0.0481, "reason": "content_based(rank=3) | matrix_factorization(rank=1) | graph_rwr(rank=3)"},
    {"name": "Breast Neoplasm",  "score": 0.0479, "reason": "content_based(rank=1) | matrix_factorization(rank=6) | graph_rwr(rank=1)"}
  ]
}
```

### Gene recommendations for a disease
```bash
curl "http://localhost:8000/api/v1/diseases/Breast%20Neoplasm/recommend?top_k=5"
```

### Similar genes
```bash
curl "http://localhost:8000/api/v1/genes/TP53/similar?top_k=5"
```

### Similar diseases
```bash
curl "http://localhost:8000/api/v1/diseases/Cardiomyopathy/similar?top_k=5"
```

### Network graph (D3-compatible)
```bash
curl "http://localhost:8000/api/v1/genes/KRAS/network?depth=2&max_nodes=40"
```

### Switch to a specific model
```bash
# options: hybrid_rrf | content_based | matrix_factorization | graph_rwr
curl "http://localhost:8000/api/v1/genes/EGFR/recommend?model_name=graph_rwr&top_k=5"
```

### Dataset statistics
```bash
curl "http://localhost:8000/api/v1/stats"
```

### Interactive Swagger UI
Open **http://localhost:8000/docs** in your browser to explore every endpoint
with a built-in form interface.

---

## Web UI

Start both services and open `http://localhost:5173`:

```bash
# Terminal 1 – backend
python main.py --serve

# Terminal 2 – frontend
cd frontend
npm install     # first time only
npm run dev
```

| Page | What it does |
|------|-------------|
| **Dashboard** | Dataset statistics, top genes/diseases bar charts, quick-search |
| **Gene Explorer** | Search a gene → ranked disease recommendations + similar genes |
| **Disease Explorer** | Search a disease → ranked gene candidates + similar diseases |
| **Knowledge Graph** | Interactive D3 force-directed bipartite network, drag & zoom |
| **About** | Algorithm documentation and data source references |

---

## Evaluate the recommender

Run leave-one-out cross-validation and print IR metrics:

```bash
python main.py --evaluate
```

```json
{
  "k": 10,
  "n_gene_queries": 200,
  "gene": {
    "precision_at_k": 0.42,
    "recall_at_k":    0.31,
    "ndcg_at_k":      0.48,
    "mrr":            0.55,
    "map_at_k":       0.44,
    "hit_rate_at_k":  0.73
  }
}
```

---

## Run tests

```bash
pytest tests/ -v
```
