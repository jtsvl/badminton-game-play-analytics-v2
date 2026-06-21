import { describe, it, expect, beforeAll, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { PlayerReviewBoard } from './PlayerReviewBoard'
import type { ReviewPlayerOption, ReviewRow } from './types'
import type { TrackKey } from '../../lib/overlay'

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

// Identity homography: court (u,v) maps straight to frame fraction → projectUV is well-defined in jsdom.
const H_ID = [1, 0, 0, 0, 1, 0, 0, 0, 1]
const SIZE = { w: 1920, h: 1080 }

const players: ReviewPlayerOption[] = [
  { id: 'p1', name: 'Lukas Muster' },
  { id: 'p2', name: 'Jonas Beispiel' },
]
const rows: ReviewRow[] = [
  { id: 'mp-a', side: 'A', slot: 0, playerId: 'p1', trackLabel: 5 },
  { id: 'mp-b', side: 'B', slot: 0, playerId: 'p2', trackLabel: 8 },
]
const path: TrackKey[] = [
  { t: 0, u: 0.2, v: 0.8 },
  { t: 1, u: 0.5, v: 0.6 },
]
const pathByLabel = new Map<number, TrackKey[]>([
  [5, path],
  [8, path],
])

describe('PlayerReviewBoard', () => {
  it('renders the slots, player names and track options for a populated fixture', () => {
    render(
      <PlayerReviewBoard
        initialRows={rows}
        players={players}
        pathByLabel={pathByLabel}
        H={H_ID}
        frameUrl="blob:frame"
        size={SIZE}
        discipline="singles"
        saving={false}
        saveError={null}
        onConfirm={() => {}}
        onSave={() => {}}
      />,
    )
    expect(screen.getByText('Feld A')).toBeInTheDocument()
    expect(screen.getByText('Feld B')).toBeInTheDocument()
    // Both player options exist in the dropdowns.
    expect(screen.getAllByRole('option', { name: 'Lukas Muster' }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('option', { name: 'Spur #5' }).length).toBeGreaterThan(0)
    // No change yet → Bestätigen enabled, Speichern disabled.
    expect(screen.getByRole('button', { name: 'Bestätigen' })).toBeEnabled()
    expect(screen.getByRole('button', { name: /Speichern/ })).toBeDisabled()
  })

  it('confirms without changes (no recompute)', () => {
    const onConfirm = vi.fn()
    const onSave = vi.fn()
    render(
      <PlayerReviewBoard
        initialRows={rows}
        players={players}
        pathByLabel={pathByLabel}
        H={H_ID}
        frameUrl={null}
        size={SIZE}
        discipline="singles"
        saving={false}
        saveError={null}
        onConfirm={onConfirm}
        onSave={onSave}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Bestätigen' }))
    expect(onConfirm).toHaveBeenCalledTimes(1)
    expect(onSave).not.toHaveBeenCalled()
  })

  it('enables Save and emits only the changed rows after a track swap', () => {
    const onSave = vi.fn()
    render(
      <PlayerReviewBoard
        initialRows={rows}
        players={players}
        pathByLabel={pathByLabel}
        H={H_ID}
        frameUrl={null}
        size={SIZE}
        discipline="singles"
        saving={false}
        saveError={null}
        onConfirm={() => {}}
        onSave={onSave}
      />,
    )
    // The 4 selects are [A:player, A:track, B:player, B:track]; swap A's track 5 → 8 then B's 8 → 5.
    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[]
    fireEvent.change(selects[1], { target: { value: '8' } }) // A track → 8
    fireEvent.change(selects[3], { target: { value: '5' } }) // B track → 5
    const saveBtn = screen.getByRole('button', { name: /Speichern/ })
    expect(saveBtn).toBeEnabled()
    fireEvent.click(saveBtn)
    expect(onSave).toHaveBeenCalledTimes(1)
    const changed = onSave.mock.calls[0][0] as ReviewRow[]
    expect(changed.map((r) => r.id).sort()).toEqual(['mp-a', 'mp-b'])
    expect(changed.find((r) => r.id === 'mp-a')?.trackLabel).toBe(8)
  })

  it('blocks Save and warns when a player is assigned twice', () => {
    render(
      <PlayerReviewBoard
        initialRows={rows}
        players={players}
        pathByLabel={pathByLabel}
        H={H_ID}
        frameUrl={null}
        size={SIZE}
        discipline="singles"
        saving={false}
        saveError={null}
        onConfirm={() => {}}
        onSave={() => {}}
      />,
    )
    const selects = screen.getAllByRole('combobox') as HTMLSelectElement[]
    fireEvent.change(selects[2], { target: { value: 'p1' } }) // B's player → p1 (now duplicated)
    expect(screen.getByText(/mehrfach zugeordnet/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Speichern/ })).toBeDisabled()
  })

  it('renders an empty/null fixture (no tracks, no calibration) without crashing', () => {
    const emptyRows: ReviewRow[] = [{ id: 'mp-a', side: 'A', slot: 0, playerId: 'p1', trackLabel: null }]
    render(
      <PlayerReviewBoard
        initialRows={emptyRows}
        players={players}
        pathByLabel={new Map()}
        H={null}
        frameUrl={null}
        size={{ w: 0, h: 0 }}
        discipline="singles"
        saving={false}
        saveError={null}
        onConfirm={() => {}}
        onSave={() => {}}
      />,
    )
    expect(screen.getByText('Feld A')).toBeInTheDocument()
    expect(screen.getByText(/Keine Kalibrierung/)).toBeInTheDocument()
  })
})
