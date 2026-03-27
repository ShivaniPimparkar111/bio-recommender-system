"""
Disease endpoints  –  simple, forgiving, well-documented.

All disease lookups are case-insensitive ("breast cancer" == "Breast Cancer").
On 404, the response includes close-match suggestions.

GET /api/v1/diseases                         – paginated / searchable disease list
GET /api/v1/diseases/{disease}/similar       – diseases with similar gene profiles
GET /api/v1/diseases/{disease}/recommend     – top gene recommendations
GET /api/v1/diseases/{disease}/network       – D3-compatible bipartite sub-graph
GET /api/v1/diseases/{disease}               – disease detail + associated genes
"""

from __future__ import annotations

import difflib

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["diseases"])


# ── List ─────────────────────────────────────────────────────────────────────

@router.get("/diseases", summary="List all diseases")
async def list_diseases(
    request:   Request,
    page:      int = Query(1,   ge=1,         description="Page number"),
    page_size: int = Query(50,  ge=1, le=500, description="Results per page"),
    search:    str = Query("",                description="Filter by substring (case-insensitive)"),
):
    """Return a paginated list of disease names. Use `search` to filter."""
    diseases = request.app.state.model.diseases

    if search:
        s = search.lower()
        diseases = [d for d in diseases if s in d.lower()]

    total = len(diseases)
    start = (page - 1) * page_size
    return {
        "total":     total,
        "page":      page,
        "page_size": page_size,
        "results":   diseases[start : start + page_size],
    }


# ── Action routes BEFORE the bare detail route ───────────────────────────────

@router.get("/diseases/{disease}/recommend", summary="Gene recommendations for a disease")
async def recommend_genes(
    request:    Request,
    disease:    str,
    top_k:      int = Query(10, ge=1, le=50),
    model_name: str = Query("hybrid_rrf"),
):
    """
    Return the top-K genes most likely to be causally associated with **disease**.

    Pass the disease name URL-encoded, e.g. `Breast%20Neoplasm`.
    The lookup is case-insensitive.
    """
    model   = _resolve_model(request, model_name)
    disease = _resolve_disease(request, disease)

    resp = model.recommend_for_disease(disease, top_k=top_k)
    if not resp.results:
        raise HTTPException(404, f"No recommendations found for disease '{disease}'.")
    return {"query": resp.query, "model": resp.model,
            "results": [r.__dict__ for r in resp.results]}


@router.get("/diseases/{disease}/similar", summary="Diseases similar to the query disease")
async def similar_diseases(
    request:    Request,
    disease:    str,
    top_k:      int = Query(10, ge=1, le=50),
    model_name: str = Query("hybrid_rrf"),
):
    """Return diseases whose gene profile is most similar to **disease**."""
    model   = _resolve_model(request, model_name)
    disease = _resolve_disease(request, disease)

    resp = model.similar_diseases(disease, top_k=top_k)
    if not resp.results:
        raise HTTPException(404, f"No similar diseases found for '{disease}'.")
    return {"query": resp.query, "model": resp.model,
            "results": [r.__dict__ for r in resp.results]}


@router.get("/diseases/{disease}/network", summary="Network sub-graph for a disease")
async def disease_network(
    request:   Request,
    disease:   str,
    depth:     int = Query(2, ge=1, le=3),
    max_nodes: int = Query(80, ge=10, le=200),
):
    """Return a D3-compatible node-link graph centred on **disease**."""
    disease = _resolve_disease(request, disease)
    data    = request.app.state.model.get_network_data(disease, depth=depth, max_nodes=max_nodes)
    if not data["nodes"]:
        raise HTTPException(404, f"Disease '{disease}' not found in network.")
    return data


@router.get("/diseases/{disease}", summary="Disease detail")
async def get_disease(request: Request, disease: str):
    """Return a disease's metadata and all known associated genes."""
    disease = _resolve_disease(request, disease)
    df      = request.app.state.df
    sub     = df[df["disease"] == disease]
    return {
        "disease":    disease,
        "gene_count": int(len(sub)),
        "genes":      sorted(sub["gene"].unique().tolist()),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _resolve_disease(request: Request, raw: str) -> str:
    """
    Case-insensitive lookup: tries title-case first, then full case-fold scan.
    On mismatch returns a 404 with close-match suggestions.
    """
    diseases   = request.app.state.model.diseases
    lower_map  = {d.lower(): d for d in diseases}  # case-fold lookup table
    normalised = raw.strip()

    # 1. Exact match
    if normalised in diseases:
        return normalised

    # 2. Case-insensitive match
    key = normalised.lower()
    if key in lower_map:
        return lower_map[key]

    # 3. Title-case match
    titled = normalised.title()
    if titled in diseases:
        return titled

    # 4. Word-token partial match — find diseases sharing at least one word
    #    e.g. "Breast Cancer" → matches "Breast Neoplasm", "Malignant Tumor Of Breast"
    query_words = {w for w in key.split() if len(w) > 2}
    token_matches = [
        d for d in diseases
        if any(w in d.lower() for w in query_words)
    ][:5]

    # 5. Difflib character-level suggestions as fallback
    char_matches = difflib.get_close_matches(titled, diseases, n=5, cutoff=0.4)

    suggestions = token_matches or char_matches
    detail = f"Disease '{normalised}' not found."
    if suggestions:
        detail += f" Did you mean: {', '.join(suggestions)}?"
    raise HTTPException(status_code=404, detail=detail)


def _resolve_model(request: Request, model_name: str):
    hybrid = request.app.state.model
    mapping = {
        "content_based":        hybrid._cb,
        "matrix_factorization": hybrid._mf,
        "graph_rwr":            hybrid._rwr,
    }
    return mapping.get(model_name, hybrid)
