import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { MatchList } from './pages/MatchList'
import { MatchReport } from './pages/MatchReport'
import { Calibrate } from './pages/Calibrate'
import { PlayerReview } from './pages/PlayerReview'
import { RallyReview } from './pages/RallyReview'
import { Upload } from './pages/Upload'

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MatchList />} />
        <Route path="/match/:slug" element={<MatchReport />} />
        <Route path="/match/:slug/calibrate" element={<Calibrate />} />
        <Route path="/match/:slug/review" element={<PlayerReview />} />
        <Route path="/match/:slug/rallies" element={<RallyReview />} />
        <Route path="/upload" element={<Upload />} />
      </Routes>
    </BrowserRouter>
  )
}
