import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { HitTimeline } from './HitTimeline'
import type { HitReviewRow } from './hitPlan'

afterEach(cleanup)

function hit(id: string, tS: number, over: Partial<HitReviewRow> = {}): HitReviewRow {
  return { id, tS, rallyId: 'r1', rallyIdx: 1, confidence: 0.3, payload: null, ...over }
}

function handlers() {
  return {
    onSelect: vi.fn(), onSeek: vi.fn(), onAddAtPlayhead: vi.fn(),
    onNudge: vi.fn(), onDelete: vi.fn(),
  }
}

describe('HitTimeline', () => {
  it('renders one tick per hit', () => {
    const h = handlers()
    render(<HitTimeline hits={[hit('a', 10), hit('b', 20)]} dur={100} time={0} fps={60} selectedId={null} {...h} />)
    expect(screen.getAllByRole('button', { name: /Schlag bei/ })).toHaveLength(2)
    expect(screen.getByText(/2 Schläge/)).toBeInTheDocument()
  })

  it('clicking a tick selects it and seeks to its time', () => {
    const h = handlers()
    render(<HitTimeline hits={[hit('a', 10), hit('b', 20)]} dur={100} time={0} fps={60} selectedId={null} {...h} />)
    fireEvent.pointerDown(screen.getByRole('button', { name: /Schlag bei 0:20/ }))
    expect(h.onSelect).toHaveBeenCalledWith('b')
    expect(h.onSeek).toHaveBeenCalledWith(20)
  })

  it('the add button adds a hit at the playhead', () => {
    const h = handlers()
    render(<HitTimeline hits={[hit('a', 10)]} dur={100} time={0} fps={60} selectedId={null} {...h} />)
    fireEvent.click(screen.getByRole('button', { name: /Schlag hier/ }))
    expect(h.onAddAtPlayhead).toHaveBeenCalledOnce()
  })

  it('with a selection, frame-nudge and delete fire for that hit', () => {
    const h = handlers()
    render(<HitTimeline hits={[hit('a', 10)]} dur={100} time={0} fps={50} selectedId="a" {...h} />)
    fireEvent.click(screen.getByRole('button', { name: /1 Frame vor/ }))
    expect(h.onNudge).toHaveBeenCalledWith('a', 10 + 1 / 50)
    fireEvent.click(screen.getByRole('button', { name: /Löschen/ }))
    expect(h.onDelete).toHaveBeenCalledWith('a')
  })

  it('added hits are visually distinct from draft hits', () => {
    const h = handlers()
    render(<HitTimeline hits={[hit('a', 10), hit('tmp-1', 30, { isNew: true })]} dur={100} time={0} fps={60} selectedId={null} {...h} />)
    const ticks = screen.getAllByRole('button', { name: /Schlag bei/ })
    const colors = ticks.map((t) => t.style.background)
    expect(new Set(colors).size).toBe(2) // draft orange vs added blue
  })
})
