import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { VideoControls } from './VideoControls'
import type { VideoPlayer } from '../../hooks/useVideoPlayer'

afterEach(cleanup)

function mockPlayer(over: Partial<VideoPlayer> = {}): VideoPlayer {
  return {
    vref: { current: null },
    playing: false, setPlaying: vi.fn(), time: 10, setTime: vi.fn(), dur: 100, hasFootage: true,
    ov: { tracking: true, wege: true, shuttle: false, zonen: false, skelett: false }, toggle: vi.fn(),
    onLoaded: vi.fn(), togglePlay: vi.fn(), seek: vi.fn(), rate: 1, setRate: vi.fn(),
    ...over,
  } as VideoPlayer
}

describe('VideoControls', () => {
  it('shows no speed control without showRate (opt-in preserved for the report)', () => {
    render(<VideoControls player={mockPlayer()} />)
    expect(screen.queryByText('Tempo')).toBeNull()
  })

  it('with showRate, a speed button calls setRate and the active rate is marked', () => {
    const p = mockPlayer({ rate: 0.5 })
    render(<VideoControls player={p} showRate fps={60} />)
    expect(screen.getByRole('button', { name: 'Tempo 0,5×' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Tempo 1×' })).toHaveAttribute('aria-pressed', 'false')
    fireEvent.click(screen.getByRole('button', { name: 'Tempo 0,25×' }))
    expect(p.setRate).toHaveBeenCalledWith(0.25)
  })

  it('frame-step forward pauses if playing and seeks by exactly 1/fps', () => {
    const p = mockPlayer({ playing: true, time: 10, dur: 100 })
    render(<VideoControls player={p} showRate fps={50} />)
    fireEvent.click(screen.getByRole('button', { name: 'Video 1 Frame vor' }))
    expect(p.togglePlay).toHaveBeenCalledOnce() // paused before stepping
    expect(p.seek).toHaveBeenCalledWith(10 + 1 / 50)
  })

  it('frame-step back clamps at 0 and does not re-pause when already paused', () => {
    const p = mockPlayer({ playing: false, time: 0, dur: 100 })
    render(<VideoControls player={p} showRate fps={50} />)
    fireEvent.click(screen.getByRole('button', { name: 'Video 1 Frame zurück' }))
    expect(p.togglePlay).not.toHaveBeenCalled()
    expect(p.seek).toHaveBeenCalledWith(0)
  })
})
