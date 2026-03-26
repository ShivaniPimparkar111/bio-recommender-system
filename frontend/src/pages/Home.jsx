import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Dna, Network, BarChart2, Sparkles } from 'lucide-react'
import { getStats } from '../services/api'
import StatsPanel from '../components/StatsPanel'
import SearchBar from '../components/SearchBar'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

export default function Home() {
  const [stats, setStats]   = useState(null)
  const navigate             = useNavigate()

  useEffect(() => {
    getStats().then(setStats).catch(console.error)
  }, [])

  const handleSearch = (value, type) => {
    if (type === 'gene')    navigate(`/genes?q=${encodeURIComponent(value)}`)
    else                    navigate(`/diseases?q=${encodeURIComponent(value)}`)
  }

  // Prepare top-genes chart data
  const topGenesData = stats
    ? Object.entries(stats.top_genes || {})
        .slice(0, 8)
        .map(([gene, count]) => ({ gene, count }))
    : []

  const topDiseasesData = stats
    ? Object.entries(stats.top_diseases || {})
        .slice(0, 6)
        .map(([disease, count]) => ({ disease: disease.slice(0, 22), count }))
    : []

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12"
    >
      {/* ── Hero ── */}
      <section className="text-center space-y-6 py-8">
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1,   opacity: 1 }}
          transition={{ type: 'spring', stiffness: 200 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
                     bg-brand-900/50 border border-brand-800 text-brand-400 text-sm mb-2"
        >
          <Sparkles size={14} />
          Computational Biology
        </motion.div>

        <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white leading-tight">
          Gene–Disease
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-purple-400">
            {' '}Recommender
          </span>
        </h1>

        <p className="text-lg text-slate-400 max-w-2xl mx-auto">
          Combining <strong className="text-slate-300">TF-IDF content-based filtering</strong>,{' '}
          <strong className="text-slate-300">NMF matrix factorisation</strong>, and{' '}
          <strong className="text-slate-300">bipartite graph random walks</strong>{' '}
          over curated NCBI ClinVar associations.
        </p>

        {/* Search */}
        <div className="max-w-xl mx-auto">
          <SearchBar type="both" onSelect={handleSearch} placeholder="Search any gene or disease…" />
        </div>

        {/* Quick-links */}
        <div className="flex flex-wrap items-center justify-center gap-3 pt-2">
          {['BRCA1', 'TP53', 'EGFR', 'KRAS', 'ATM'].map(g => (
            <button
              key={g}
              onClick={() => navigate(`/genes?q=${g}`)}
              className="tag-gene hover:bg-brand-800/80 transition-colors cursor-pointer"
            >
              {g}
            </button>
          ))}
          {['Breast Cancer', 'Lung Cancer', 'Autism'].map(d => (
            <button
              key={d}
              onClick={() => navigate(`/diseases?q=${encodeURIComponent(d)}`)}
              className="tag-disease hover:bg-purple-800/60 transition-colors cursor-pointer"
            >
              {d}
            </button>
          ))}
        </div>
      </section>

      {/* ── Stats ── */}
      {stats && <StatsPanel stats={stats} />}

      {/* ── Charts ── */}
      {topGenesData.length > 0 && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Top genes */}
          <div className="card space-y-4">
            <p className="section-header">Top Genes by Disease Count</p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={topGenesData} layout="vertical" margin={{ left: 12, right: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3560" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis dataKey="gene" type="category" tick={{ fill: '#cbd5e1', fontSize: 11 }} width={60} />
                <Tooltip
                  contentStyle={{ background: '#1a2340', border: '1px solid #2a3560', borderRadius: 8 }}
                  labelStyle={{ color: '#e2e8f0' }}
                  itemStyle={{ color: '#52b4fd' }}
                />
                <Bar dataKey="count" fill="#2993fa" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top diseases */}
          <div className="card space-y-4">
            <p className="section-header">Top Diseases by Gene Count</p>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={topDiseasesData} layout="vertical" margin={{ left: 16, right: 24 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a3560" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis dataKey="disease" type="category" tick={{ fill: '#cbd5e1', fontSize: 10 }} width={130} />
                <Tooltip
                  contentStyle={{ background: '#1a2340', border: '1px solid #2a3560', borderRadius: 8 }}
                  labelStyle={{ color: '#e2e8f0' }}
                  itemStyle={{ color: '#c084fc' }}
                />
                <Bar dataKey="count" fill="#9333ea" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Feature cards ── */}
      <section className="grid sm:grid-cols-3 gap-5">
        {[
          {
            icon: Dna, title: 'Content-Based', color: 'text-brand-400', bg: 'bg-brand-900/30',
            desc: 'TF-IDF vectorisation of gene/disease profiles with cosine-similarity ranking.',
          },
          {
            icon: BarChart2, title: 'Matrix Factorisation', color: 'text-purple-400', bg: 'bg-purple-900/30',
            desc: 'Non-negative Matrix Factorisation learns latent biological pathway themes.',
          },
          {
            icon: Network, title: 'Graph RWR', color: 'text-emerald-400', bg: 'bg-emerald-900/30',
            desc: 'Personalised PageRank over a bipartite gene–disease knowledge graph.',
          },
        ].map(({ icon: Icon, title, color, bg, desc }, i) => (
          <motion.div
            key={title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + i * 0.1 }}
            className="card space-y-3"
          >
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${bg}`}>
              <Icon className={`w-5 h-5 ${color}`} />
            </div>
            <h3 className="font-semibold text-white">{title}</h3>
            <p className="text-sm text-slate-400">{desc}</p>
          </motion.div>
        ))}
      </section>
    </motion.div>
  )
}
