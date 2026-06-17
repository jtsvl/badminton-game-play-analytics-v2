import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { MatchList } from './pages/MatchList'
import { MatchReport } from './pages/MatchReport'
import { Upload } from './pages/Upload'

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MatchList />} />
        <Route path="/match/:slug" element={<MatchReport />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </BrowserRouter>
  )
}
