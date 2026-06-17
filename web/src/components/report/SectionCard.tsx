import type { ReactNode } from 'react'

interface Props {
  title: string
  sub?: string
  dense?: boolean
  children: ReactNode
}

/** Brand card wrapper (specs/50-web-app.md). `dense` tightens padding for the desktop tiles. */
export function SectionCard({ title, sub, dense, children }: Props) {
  return (
    <section
      style={{
        background: 'var(--card)', border: '1px solid var(--hairline)',
        borderRadius: 'var(--radius-card)', boxShadow: 'var(--shadow-sm)',
        padding: dense ? '14px 16px 16px' : '15px 16px 18px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 10 }}>
        <h3 style={{ fontSize: dense ? 14 : 15, fontWeight: 700 }}>{title}</h3>
        {sub && <span style={{ fontSize: 11.5, color: 'var(--muted)' }}>{sub}</span>}
      </div>
      {children}
    </section>
  )
}
