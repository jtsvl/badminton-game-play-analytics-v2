import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { RallyReviewBoard } from './RallyReviewBoard'
import type { RallyReviewRow, RallySavePlan } from './rallyPlan'

afterEach(cleanup)

function row(id: string, idx: number, over: Partial<RallyReviewRow> = {}): RallyReviewRow {
  return {
    id,
    idx,
    tStart: idx * 60,
    tEnd: idx * 60 + 30,
    shotCount: 4,
    winnerSide: null,
    endShotType: null,
    confidence: 0.8,
    editedAt: null,
    ...over,
  }
}

function renderBoard(rows: RallyReviewRow[], onSave: (p: RallySavePlan) => void = () => {}) {
  return render(<RallyReviewBoard initialRows={rows} saving={false} saveError={null} onSave={onSave} />)
}

describe('RallyReviewBoard', () => {
  it('renders spans, duration, shots and confidence in de-DE', () => {
    renderBoard([row('a', 0, { tStart: 65, tEnd: 95, shotCount: 7, confidence: 0.85 })])
    expect(screen.getByText(/1:05\s*–\s*1:35/)).toBeInTheDocument()
    expect(screen.getByText(/0:30/)).toBeInTheDocument() // duration
    expect(screen.getByText(/7 Schläge/)).toBeInTheDocument()
    expect(screen.getByText(/85\s*%/)).toBeInTheDocument()
  })

  it('marks already-edited rallies', () => {
    renderBoard([row('a', 0, { editedAt: '2026-07-01T10:00:00Z' })])
    expect(screen.getByText('bearbeitet')).toBeInTheDocument()
  })

  it('disables Save until something is flagged', () => {
    renderBoard([row('a', 0), row('b', 1)])
    expect(screen.getByRole('button', { name: /Speichern/ })).toBeDisabled()
  })

  it('offers no merge toggle on the first rally', () => {
    renderBoard([row('a', 0), row('b', 1)])
    expect(screen.queryByRole('checkbox', { name: /Ballwechsel 1 mit vorherigem/ })).toBeNull()
    expect(screen.getByRole('checkbox', { name: /Ballwechsel 2 mit vorherigem/ })).toBeInTheDocument()
  })

  it('emits a drop plan', () => {
    const onSave = vi.fn()
    renderBoard([row('a', 0), row('b', 1)], onSave)
    fireEvent.click(screen.getByRole('checkbox', { name: /Ballwechsel 2 entfernen/ }))
    const saveBtn = screen.getByRole('button', { name: /Speichern/ })
    expect(saveBtn).toBeEnabled()
    fireEvent.click(saveBtn)
    const plan = onSave.mock.calls[0][0] as RallySavePlan
    expect(plan.rallyDeletes).toEqual([{ id: 'b', deleteEvents: true }])
    expect(plan.rallyUpdates).toEqual([])
  })

  it('emits a merge plan with the folded patch', () => {
    const onSave = vi.fn()
    renderBoard(
      [row('a', 0, { tEnd: 30, shotCount: 2, confidence: 0.9 }), row('b', 1, { tEnd: 95, shotCount: 3, confidence: 0.7, endShotType: 'smash' })],
      onSave,
    )
    fireEvent.click(screen.getByRole('checkbox', { name: /Ballwechsel 2 mit vorherigem/ }))
    fireEvent.click(screen.getByRole('button', { name: /Speichern/ }))
    const plan = onSave.mock.calls[0][0] as RallySavePlan
    expect(plan.rallyUpdates).toHaveLength(1)
    expect(plan.rallyUpdates[0].id).toBe('a')
    expect(plan.rallyUpdates[0].patch.t_end_s).toBe(95)
    expect(plan.rallyUpdates[0].patch.shot_count).toBe(5)
    expect(plan.rallyUpdates[0].patch.confidence).toBe(0.7)
    expect(plan.eventReparents).toEqual([{ from: 'b', to: 'a' }])
  })

  it('blocks Save and warns when merging next to a dropped rally', () => {
    renderBoard([row('a', 0), row('b', 1), row('c', 2)])
    fireEvent.click(screen.getByRole('checkbox', { name: /Ballwechsel 2 entfernen/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: /Ballwechsel 3 mit vorherigem/ }))
    expect(screen.getByText(/entfernten Ballwechsel/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Speichern/ })).toBeDisabled()
  })

  it('shows a save error from the container', () => {
    render(<RallyReviewBoard initialRows={[row('a', 0)]} saving={false} saveError="RLS sagt nein" onSave={() => {}} />)
    expect(screen.getByText('RLS sagt nein')).toBeInTheDocument()
  })

  // Video wiring (video-proxy spec): the container owns the player; the board only exposes a
  // per-row "Ansehen" seek affordance and highlights the row under the playhead.
  it('renders no play buttons without onPlayRally (list-only fallback)', () => {
    renderBoard([row('a', 0), row('b', 1)])
    expect(screen.queryByRole('button', { name: /ansehen/i })).toBeNull()
  })

  it('calls onPlayRally with the row when its play button is clicked', () => {
    const onPlay = vi.fn()
    render(
      <RallyReviewBoard
        initialRows={[row('a', 0), row('b', 1)]}
        saving={false}
        saveError={null}
        onSave={() => {}}
        onPlayRally={onPlay}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: /Ballwechsel 2 ansehen/i }))
    expect(onPlay).toHaveBeenCalledTimes(1)
    expect((onPlay.mock.calls[0][0] as RallyReviewRow).id).toBe('b')
  })

  it('marks the active rally row via aria-current', () => {
    render(
      <RallyReviewBoard
        initialRows={[row('a', 0), row('b', 1)]}
        saving={false}
        saveError={null}
        onSave={() => {}}
        onPlayRally={() => {}}
        activeRallyId="b"
      />,
    )
    const active = document.querySelectorAll('[aria-current="true"]')
    expect(active).toHaveLength(1)
    expect(active[0].textContent).toContain('Ballwechsel 2')
  })
})
