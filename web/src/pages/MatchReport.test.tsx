import { describe, it, expect, beforeAll } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach } from 'vitest'
import { MatchReportMobile } from './MatchReportMobile'
import { MatchReportDashboard } from './MatchReportDashboard'
import type { MatchReport, ReportPlayer, ShotPair, Side } from '../types'

// Reduced motion → CountUp/Reveal short-circuit (no rAF/IO needed in jsdom).
beforeAll(() => {
  window.matchMedia = ((q: string) => ({
    matches: q.includes('reduce'),
    media: q,
    onchange: null,
    addEventListener() {},
    removeEventListener() {},
    addListener() {},
    removeListener() {},
    dispatchEvent() {
      return false
    },
  })) as unknown as typeof window.matchMedia
})

afterEach(cleanup)

function mkPlayer(over: { side: Side; first: string; name?: string; isViewer?: boolean; shots?: ShotPair[] }): ReportPlayer {
  return {
    playerId: `p-${over.first}`,
    name: over.name ?? `${over.first} Muster`,
    first: over.first,
    isViewer: over.isViewer ?? false,
    side: over.side,
    stats: {
      distance_m: 2840,
      avg_kmh: 5.9,
      top_kmh: 21.4,
      sprints: 27,
      zones: { front: 31, mid: 44, rear: 25, left: 48, right: 52 },
      timeline: [225, 300, 330, 275, 290, 340, 250, 280, 310, 240],
      heat: [{ x: 0.5, y: 0.55, w: 1, s: 0.18 }],
      heatmap_uri: null,
      fastest: 196,
      shots: over.shots ?? [
        ['clear', 84],
        ['smash', 38],
      ],
    },
  }
}

const singles: MatchReport = {
  video: { src: '', label: 'Demo-Aufnahme', durationS: null },
  calib: null,
  tracks: [],
  meta: {
    id: '1',
    dateLabel: '10. Juni 2026',
    weekday: 'Mittwoch',
    dd: '10.06.',
    discipline: 'singles',
    durationMin: 48,
    restMin: 14,
    court: 1,
    score: '21:18 · 19:21 · 21:15',
    games: [[21, 18], [19, 21], [21, 15]],
    result: 'win',
    resultLine: 'Sieg 2:1',
    status: 'done',
  },
  rallies: { count: 112, avgShots: 6.5, avgSec: 7.8, longShots: 34, longSec: 52, endLine: 'Der Smash beendete 31 % aller Ballwechsel.' },
  pairs: [], // hook returns [] for singles — must not be treated as doubles
  players: [mkPlayer({ side: 'A', first: 'Lukas', isViewer: true }), mkPlayer({ side: 'B', first: 'Jonas' })],
}

// Doubles: no shots, no video, with pairs — exercises every guard.
const doubles: MatchReport = {
  video: null,
  calib: null,
  tracks: [],
  meta: {
    id: '2',
    dateLabel: '3. Juni 2026',
    weekday: 'Mittwoch',
    dd: '03.06.',
    discipline: 'doubles',
    durationMin: 52,
    restMin: 10,
    court: 2,
    score: '21:17 · 21:19',
    games: [[21, 17], [21, 19]],
    result: 'win',
    resultLine: 'Sieg 2:0',
    status: 'done',
  },
  rallies: { count: 89, avgShots: 5.4, avgSec: 6.6, longShots: 28, longSec: 40, endLine: '' },
  pairs: [
    { side: 'A', label: 'Lukas & Miriam', fb: 58, ss: 42, rot: 46 },
    { side: 'B', label: 'David & Anna', fb: 41, ss: 59, rot: 31 },
  ],
  players: [
    mkPlayer({ side: 'A', first: 'Lukas', isViewer: true, shots: [] }),
    mkPlayer({ side: 'A', first: 'Miriam', shots: [] }),
    mkPlayer({ side: 'B', first: 'David', shots: [] }),
    mkPlayer({ side: 'B', first: 'Anna', shots: [] }),
  ],
}

const wrap = (ui: React.ReactNode) => render(<MemoryRouter>{ui}</MemoryRouter>)

describe('MatchReportMobile', () => {
  it('renders the singles report without crashing', () => {
    wrap(<MatchReportMobile report={singles} tab={0} setTab={() => {}} slug="2026-06-10" />)
    expect(screen.getByText('Feldabdeckung')).toBeInTheDocument()
    expect(screen.getByText('Video-Analyse')).toBeInTheDocument()
    expect(screen.getByText('Schlagverteilung')).toBeInTheDocument()
  })

  it('renders the doubles report (no shots/video) with formation and no crash', () => {
    wrap(<MatchReportMobile report={doubles} tab={0} setTab={() => {}} slug="2026-06-03" />)
    expect(screen.getByText('Doppel-Formation')).toBeInTheDocument()
    // no shots → no Schlagverteilung section on mobile; no video → no Video-Analyse
    expect(screen.queryByText('Video-Analyse')).not.toBeInTheDocument()
    expect(screen.queryByText('Schlagverteilung')).not.toBeInTheDocument()
  })
})

describe('MatchReportDashboard', () => {
  it('renders the singles dashboard without crashing', () => {
    wrap(<MatchReportDashboard report={singles} tab={0} setTab={() => {}} slug="2026-06-10" />)
    expect(screen.getByText('Endstand')).toBeInTheDocument()
    expect(screen.getByText('Kennzahlen')).toBeInTheDocument()
  })

  it('renders the doubles dashboard (no shots/video) with the video placeholder + empty shot tile', () => {
    wrap(<MatchReportDashboard report={doubles} tab={0} setTab={() => {}} slug="2026-06-03" />)
    expect(screen.getByText('Für dieses Spiel liegt keine Videoanalyse vor.')).toBeInTheDocument()
    expect(screen.getByText('Noch keine Schlagdaten.')).toBeInTheDocument()
    expect(screen.getByText('Schlagverteilung')).toBeInTheDocument() // empty-state tile keeps the 4-tile row
  })
})
