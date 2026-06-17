import type { CSSProperties, ReactNode } from 'react'

interface Props {
  title?: string
  sub?: ReactNode
  foot?: ReactNode
  dense?: boolean
  style?: CSSProperties
  children: ReactNode
}

/** Brand white card (specs/50-web-app.md). `dense` tightens padding for the desktop tiles;
 *  `style` lets a layout set grid-column / flex sizing; `foot` is the muted caption row. */
export function SectionCard({ title, sub, foot, dense, style, children }: Props) {
  return (
    <section
      style={{
        background: 'var(--card)',
        border: '1px solid var(--hairline)',
        borderRadius: 'var(--radius-card)',
        boxShadow: 'var(--shadow-sm)',
        padding: dense ? '14px 16px 16px' : '15px 16px 18px',
        boxSizing: 'border-box',
        ...style,
      }}
    >
      {title && (
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8, marginBottom: dense ? 11 : 13 }}>
          <h3 style={{ fontSize: dense ? 14 : 15, fontFamily: 'var(--font-heading)', fontWeight: 700, whiteSpace: 'nowrap' }}>{title}</h3>
          {sub && <span style={{ fontSize: 11.5, color: 'var(--muted)', textAlign: 'right' }}>{sub}</span>}
        </div>
      )}
      {children}
      {foot && <p style={{ margin: '12px 0 0', fontSize: 11.5, color: 'var(--muted)', lineHeight: 1.5 }}>{foot}</p>}
    </section>
  )
}
