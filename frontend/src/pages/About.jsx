import { motion } from 'framer-motion'
import { BookOpen, GitBranch, Database, FlaskConical, Network, BarChart2, Dna } from 'lucide-react'

const section = (title, icon, content, delay = 0) => ({ title, icon, content, delay })

const sections = [
  section('Project Overview', Dna,
    `GeneDiseaseAI is a computational biology recommender system that
    predicts gene–disease associations using three complementary algorithms combined
    via Reciprocal Rank Fusion (RRF). It is built on top of the freely available
    NCBI ClinVar gene_condition_source_id dataset.`, 0),

  section('Data Source', Database,
    `Primary: NCBI ClinVar gene_condition_source_id (no registration required).
    URL: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/gene_condition_source_id
    Fallback: Human Phenotype Ontology (HPO) genes_to_disease.txt.
    Data is downloaded programmatically, cleaned, and saved as data/gene_disease.csv.`, 0.1),

  section('Algorithm 1 — Content-Based (TF-IDF)', FlaskConical,
    `Each gene is represented as a TF-IDF document over its associated disease
    vocabulary. TF-IDF de-emphasises common "stop-word" diseases shared by many
    genes, amplifying rare, specific annotations. Cosine similarity in the
    resulting L2-normalised embedding space ranks candidates by functional overlap.
    Reference: Salton & McGill (1983), Cheng et al. (2014) BMC Genomics.`, 0.15),

  section('Algorithm 2 — Matrix Factorisation (NMF)', BarChart2,
    `The binary gene × disease interaction matrix M is factorised as M ≈ W·H via
    Non-negative Matrix Factorisation (NMF). Non-negativity produces parts-based,
    biologically interpretable latent factor representations—analogous to pathway
    activation themes. Recommendations are scored as inner products in latent space.
    Reference: Lee & Seung (1999) Nature 401.`, 0.2),

  section('Algorithm 3 — Graph RWR', Network,
    `Genes and diseases are modelled as a bipartite graph. Random Walk with Restart
    (RWR/Personalised PageRank) propagates a probability mass from a seed node,
    converging to a stationary distribution that captures global network context
    beyond direct neighbours—enabling transitive reasoning across the knowledge graph.
    Reference: Köhler et al. (2008) AJHG 82.`, 0.25),

  section('Ensemble — Reciprocal Rank Fusion', GitBranch,
    `The three model rankings are merged via RRF: score(d) = Σ_m 1/(k+rank_m(d))
    where k=60 is a smoothing constant. RRF is score-agnostic, avoiding the problem
    of calibrating heterogeneous score distributions, and empirically matches or
    exceeds Borda-count and weighted-sum fusion.
    Reference: Cormack, Clarke & Buettcher (2009) SIGIR.`, 0.3),

  section('Evaluation Metrics', BookOpen,
    `The system is evaluated using leave-one-out cross-validation with standard
    information-retrieval metrics:
    • Precision@K – fraction of top-K results that are relevant
    • Recall@K    – fraction of relevant items found in top-K
    • NDCG@K      – Normalised Discounted Cumulative Gain (position-weighted)
    • MRR         – Mean Reciprocal Rank
    • MAP@K       – Mean Average Precision
    • Hit Rate@K  – proportion of queries with ≥1 correct result`, 0.35),
]

export default function About() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-6"
    >
      <div className="text-center space-y-3 mb-12">
        <h1 className="text-4xl font-bold text-white">Methodology</h1>
        <p className="text-slate-400">Technical documentation for the GeneDiseaseAI recommender system.</p>
        <a
          href="https://github.com/ShivaniPimparkar111/biology-recommender"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-2 text-brand-400 hover:text-brand-300 text-sm transition-colors"
        >
          <GitBranch size={14} />
          View source on GitHub
        </a>
      </div>

      <div className="space-y-4">
        {sections.map(({ title, icon: Icon, content, delay }) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay }}
            className="card"
          >
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-xl bg-brand-900/40 border border-brand-800/50 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Icon className="w-5 h-5 text-brand-400" />
              </div>
              <div>
                <h2 className="font-semibold text-white mb-2">{title}</h2>
                <p className="text-slate-400 text-sm leading-relaxed whitespace-pre-line">{content}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Tech stack */}
      <div className="card mt-8">
        <p className="section-header">Technology Stack</p>
        <div className="grid sm:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-slate-300 font-semibold mb-2">Backend</p>
            <ul className="space-y-1 text-slate-400">
              <li>Python 3.11 · FastAPI · Uvicorn</li>
              <li>Pandas · NumPy · SciPy</li>
              <li>Scikit-learn (NMF, TF-IDF)</li>
              <li>NetworkX (graph algorithms)</li>
              <li>Pydantic v2</li>
            </ul>
          </div>
          <div>
            <p className="text-slate-300 font-semibold mb-2">Frontend</p>
            <ul className="space-y-1 text-slate-400">
              <li>React 18 · Vite · Tailwind CSS</li>
              <li>D3.js (force-directed graph)</li>
              <li>Recharts (bar charts)</li>
              <li>Framer Motion (animations)</li>
              <li>React Router v6</li>
            </ul>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
