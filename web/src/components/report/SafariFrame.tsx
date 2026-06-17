import type { CSSProperties, ReactNode } from 'react'

interface Props {
  url: string
  children: ReactNode
  style?: CSSProperties
  barH?: number
}

const DIM = 'rgba(10,10,11,.35)'

/** macOS Safari-style window chrome wrapping the desktop video surface. */
export function SafariFrame({ url, children, style, barH = 44 }: Props) {
  return (
    <div
      style={{
        borderRadius: 12,
        overflow: 'hidden',
        border: '1px solid var(--hairline)',
        background: 'var(--card)',
        boxShadow: 'var(--shadow-md)',
        ...style,
      }}
    >
      <div
        style={{
          height: barH,
          flex: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 12px',
          background: '#F4F1ED',
          borderBottom: '1px solid var(--hairline)',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ display: 'flex', gap: 7, flex: 'none' }}>
          <span style={{ width: 12, height: 12, borderRadius: 6, background: '#FF5F57', display: 'block' }} />
          <span style={{ width: 12, height: 12, borderRadius: 6, background: '#FEBC2E', display: 'block' }} />
          <span style={{ width: 12, height: 12, borderRadius: 6, background: '#28C840', display: 'block' }} />
        </div>
        <div style={{ display: 'flex', gap: 8, flex: 'none', marginLeft: 2 }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={DIM} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}>
            <path d="M15 6l-6 6 6 6" />
          </svg>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={DIM} strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}>
            <path d="M9 6l6 6-6 6" />
          </svg>
        </div>
        <div style={{ flex: 1, display: 'flex', justifyContent: 'center', minWidth: 0 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 6,
              height: 28,
              borderRadius: 8,
              background: 'rgba(10,10,11,.055)',
              padding: '0 12px',
              maxWidth: 380,
              width: '100%',
              boxSizing: 'border-box',
            }}
          >
            <svg width="10" height="12" viewBox="0 0 12 14" fill="none" stroke={DIM} strokeWidth="1.6" style={{ flex: 'none' }}>
              <rect x="1.5" y="6" width="9" height="6.5" rx="1.5" />
              <path d="M3.5 6V4.5a2.5 2.5 0 0 1 5 0V6" />
            </svg>
            <span style={{ fontSize: 12, color: 'rgba(10,10,11,.6)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{url}</span>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, flex: 'none' }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={DIM} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: 'block' }}>
            <path d="M12 3v11" />
            <path d="M8 6l4-3 4 3" />
            <path d="M6 10H5a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-9a1 1 0 0 0-1-1h-1" />
          </svg>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={DIM} strokeWidth="2.2" strokeLinecap="round" style={{ display: 'block' }}>
            <path d="M12 5v14M5 12h14" />
          </svg>
        </div>
      </div>
      {children}
    </div>
  )
}
