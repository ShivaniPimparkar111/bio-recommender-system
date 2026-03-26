import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import GeneSearch from './pages/GeneSearch'
import DiseaseSearch from './pages/DiseaseSearch'
import NetworkPage from './pages/Network'
import AboutPage from './pages/About'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/"         element={<Home />} />
            <Route path="/genes"    element={<GeneSearch />} />
            <Route path="/diseases" element={<DiseaseSearch />} />
            <Route path="/network"  element={<NetworkPage />} />
            <Route path="/about"    element={<AboutPage />} />
          </Routes>
        </AnimatePresence>
      </main>
      <footer className="border-t border-surface-border py-6 text-center text-sm text-slate-500">
        GeneDiseaseAI — Built by&nbsp;
        <a
          href="https://github.com/ShivaniPimparkar111"
          target="_blank"
          rel="noopener noreferrer"
          className="text-brand-400 hover:text-brand-300 transition-colors"
        >
          ShivaniPimparkar111
        </a>
        &nbsp;· NCBI ClinVar Data · Gene–Disease Recommender
      </footer>
    </div>
  )
}
