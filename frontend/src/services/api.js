import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const api = axios.create({
  baseURL: `${BASE}/api/v1`,
  timeout: 30_000,
})

// ── Stats ─────────────────────────────────────────────────────────────────────
export const getStats = () =>
  api.get('/stats').then(r => r.data)

// ── Genes ─────────────────────────────────────────────────────────────────────
export const listGenes = (page = 1, pageSize = 50, search = '') =>
  api.get('/genes', { params: { page, page_size: pageSize, search } }).then(r => r.data)

export const getGene = (gene) =>
  api.get(`/genes/${encodeURIComponent(gene)}`).then(r => r.data)

export const recommendDiseasesForGene = (gene, topK = 10, model = 'hybrid_rrf') =>
  api.get(`/genes/${encodeURIComponent(gene)}/recommend`, {
    params: { top_k: topK, model_name: model },
  }).then(r => r.data)

export const similarGenes = (gene, topK = 10, model = 'hybrid_rrf') =>
  api.get(`/genes/${encodeURIComponent(gene)}/similar`, {
    params: { top_k: topK, model_name: model },
  }).then(r => r.data)

export const geneNetwork = (gene, depth = 2, maxNodes = 80) =>
  api.get(`/genes/${encodeURIComponent(gene)}/network`, {
    params: { depth, max_nodes: maxNodes },
  }).then(r => r.data)

// ── Diseases ──────────────────────────────────────────────────────────────────
export const listDiseases = (page = 1, pageSize = 50, search = '') =>
  api.get('/diseases', { params: { page, page_size: pageSize, search } }).then(r => r.data)

export const getDisease = (disease) =>
  api.get(`/diseases/${encodeURIComponent(disease)}`).then(r => r.data)

export const recommendGenesForDisease = (disease, topK = 10, model = 'hybrid_rrf') =>
  api.get(`/diseases/${encodeURIComponent(disease)}/recommend`, {
    params: { top_k: topK, model_name: model },
  }).then(r => r.data)

export const similarDiseases = (disease, topK = 10, model = 'hybrid_rrf') =>
  api.get(`/diseases/${encodeURIComponent(disease)}/similar`, {
    params: { top_k: topK, model_name: model },
  }).then(r => r.data)

export const diseaseNetwork = (disease, depth = 2, maxNodes = 80) =>
  api.get(`/diseases/${encodeURIComponent(disease)}/network`, {
    params: { depth, max_nodes: maxNodes },
  }).then(r => r.data)

// ── Unified network ───────────────────────────────────────────────────────────
export const getNetwork = (entity, depth = 2, maxNodes = 80) =>
  api.get(`/network/${encodeURIComponent(entity)}`, {
    params: { depth, max_nodes: maxNodes },
  }).then(r => r.data)

// ── Evaluate ──────────────────────────────────────────────────────────────────
export const runEvaluation = (k = 10, nQueries = 100, model = 'hybrid_rrf') =>
  api.post('/evaluate', null, {
    params: { k, n_queries: nQueries, model_name: model },
  }).then(r => r.data)
