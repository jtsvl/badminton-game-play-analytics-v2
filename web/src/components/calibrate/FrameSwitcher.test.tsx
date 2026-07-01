import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import { FrameSwitcher } from './FrameSwitcher'

afterEach(cleanup)

// The switcher's whole contract: it only reports a new index. It never sees or mutates the court
// corners, so stepping frames can't disturb calibration (the court is static across snapshots).

describe('FrameSwitcher', () => {
  it('shows the 1-based frame counter and mm:ss timestamp', () => {
    render(<FrameSwitcher count={12} index={2} seconds={105} onIndex={() => {}} />)
    expect(screen.getByText(/Bild 3\/12 · 1:45/)).toBeTruthy()
  })

  it('steps forward and back via the buttons', () => {
    const onIndex = vi.fn()
    render(<FrameSwitcher count={12} index={5} seconds={0} onIndex={onIndex} />)
    fireEvent.click(screen.getByText('Weiter ▶'))
    fireEvent.click(screen.getByText('◀ Zurück'))
    expect(onIndex.mock.calls).toEqual([[6], [4]])
  })

  it('disables prev at the first frame and next at the last', () => {
    const { rerender } = render(<FrameSwitcher count={12} index={0} seconds={0} onIndex={() => {}} />)
    expect((screen.getByText('◀ Zurück') as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByText('Weiter ▶') as HTMLButtonElement).disabled).toBe(false)
    rerender(<FrameSwitcher count={12} index={11} seconds={0} onIndex={() => {}} />)
    expect((screen.getByText('Weiter ▶') as HTMLButtonElement).disabled).toBe(true)
  })

  it('clamps at the ends (no callback past the range)', () => {
    const onIndex = vi.fn()
    const { rerender } = render(<FrameSwitcher count={12} index={0} seconds={0} onIndex={onIndex} />)
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(onIndex).toHaveBeenCalledWith(0)
    onIndex.mockClear()
    rerender(<FrameSwitcher count={12} index={11} seconds={0} onIndex={onIndex} />)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(onIndex).toHaveBeenCalledWith(11)
  })

  it('steps with the arrow keys', () => {
    const onIndex = vi.fn()
    render(<FrameSwitcher count={12} index={5} seconds={0} onIndex={onIndex} />)
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(onIndex.mock.calls).toEqual([[6], [4]])
  })

  it('jumps to the slider value', () => {
    const onIndex = vi.fn()
    render(<FrameSwitcher count={12} index={0} seconds={0} onIndex={onIndex} />)
    fireEvent.change(screen.getByLabelText('Standbild wählen'), { target: { value: '8' } })
    expect(onIndex).toHaveBeenCalledWith(8)
  })

  it('renders nothing when there is only one frame (nothing to switch)', () => {
    const { container } = render(<FrameSwitcher count={1} index={0} seconds={0} onIndex={() => {}} />)
    expect(container.firstChild).toBeNull()
  })
})
