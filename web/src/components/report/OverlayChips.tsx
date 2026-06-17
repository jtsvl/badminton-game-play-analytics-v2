import type { OverlayToggles } from '../../hooks/useVideoPlayer'

interface ChipProps {
  label: string
  on: boolean
  onClick: () => void
}

function Chip({ label, on, onClick }: ChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        padding: '5px 12px',
        borderRadius: 99,
        fontSize: 12,
        fontWeight: 600,
        cursor: 'pointer',
        fontFamily: 'var(--font-body)',
        transition: 'background .15s ease, color .15s ease, border-color .15s ease',
        background: on ? '#FA8832' : 'var(--card)',
        color: on ? '#1A0500' : 'var(--muted)',
        border: `1px solid ${on ? '#C66C27' : 'var(--hairline)'}`,
        boxShadow: on ? '0 2px 0 #C66C27' : 'none',
      }}
    >
      {label}
    </button>
  )
}

interface Props {
  ov: OverlayToggles
  toggle: (key: keyof OverlayToggles) => void
}

/** Overlay layer toggles: Tracking / Laufwege / Shuttle / Zonen. */
export function OverlayChips({ ov, toggle }: Props) {
  return (
    <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
      <Chip label="Tracking" on={ov.tracking} onClick={() => toggle('tracking')} />
      <Chip label="Laufwege" on={ov.wege} onClick={() => toggle('wege')} />
      <Chip label="Shuttle" on={ov.shuttle} onClick={() => toggle('shuttle')} />
      <Chip label="Zonen" on={ov.zonen} onClick={() => toggle('zonen')} />
    </div>
  )
}
