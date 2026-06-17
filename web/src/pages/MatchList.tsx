import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { supabase, isConfigured } from '../lib/supabase'
import { dateParts } from '../lib/format'
import { disciplineDE, resultDE } from '../lib/labels'
import type { MatchListItem, MatchResultCode, Side } from '../types'

function namesForSide(players: any[], side: Side): string {
  return players
    .filter((p) => p.side === side)
    .sort((a, b) => a.slot - b.slot)
    .map((p) => (Array.isArray(p.players) ? p.players[0]?.name : p.players?.name) ?? '?')
    .join(' / ')
}

function buildItems(rows: any[]): MatchListItem[] {
  return rows.map((m, i) => {
    const dp = dateParts(m.played_on)
    const mps = m.match_players ?? []
    return {
      id: m.id, slug: m.slug, dd: dp.dd, day: dp.day, month: dp.month, dateLabel: dp.dateLabel,
      discipline: m.discipline, a: namesForSide(mps, 'A'), b: namesForSide(mps, 'B'),
      score: m.score, result: (m.result_side_a as MatchResultCode) ?? null, isNew: i === 0,
    }
  })
}

export function MatchList() {
  const [items, setItems] = useState<MatchListItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!isConfigured) { setError('Supabase ist nicht konfiguriert (.env).'); return }
    ;(async () => {
      const { data, error } = await supabase
        .from('matches')
        .select('id, slug, played_on, discipline, score, result_side_a, match_players(side, slot, players(name))')
        .order('played_on', { ascending: false })
      if (error) { setError(error.message); return }
      setItems(buildItems((data ?? []) as any[]))
    })()
  }, [])

  return (
    <main style={{ maxWidth: 480, margin: '0 auto', padding: '8px 14px 28px' }}>
      <header style={{ padding: '6px 4px 10px' }}>
        <h1 style={{ fontSize: 30, fontWeight: 700 }}>Spielberichte</h1>
        <p style={{ color: 'var(--muted)', fontSize: 13.5, margin: '4px 0 0' }}>
          Deine Auswertungen nach jedem Spiel.
        </p>
      </header>

      {error && <p style={{ color: 'var(--muted)', fontSize: 13 }}>{error}</p>}
      {!error && !items && <p style={{ color: 'var(--muted)', fontSize: 13 }}>Lädt…</p>}
      {items && items.length === 0 && (
        <p style={{ color: 'var(--muted)', fontSize: 13 }}>Noch keine ausgewerteten Spiele.</p>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {items?.map((it) => (
          <Link
            key={it.id}
            to={`/match/${it.slug}`}
            style={{
              display: 'grid', gridTemplateColumns: '46px 1fr 18px', gap: 12, alignItems: 'center',
              background: 'var(--card)', border: '1px solid var(--hairline)', borderRadius: 12,
              boxShadow: 'var(--shadow-sm)', padding: '13px 14px', position: 'relative',
            }}
          >
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: 'var(--font-head)', fontWeight: 700, fontSize: 21 }}>{it.day}</div>
              <div style={{ fontSize: 10.5, textTransform: 'uppercase', letterSpacing: '.08em', color: 'var(--muted)' }}>{it.month}</div>
            </div>
            <div style={{ minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: 14, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{it.a}</div>
              <div style={{ fontSize: 12.5, color: 'var(--muted)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>gegen {it.b}</div>
              <div style={{ display: 'flex', gap: 6, alignItems: 'center', marginTop: 4 }}>
                <span style={{ fontSize: 11, fontWeight: 600, background: 'rgba(10,10,11,.06)', borderRadius: 99, padding: '2px 8px' }}>{disciplineDE(it.discipline)}</span>
                <span style={{
                  fontSize: 11, fontWeight: 700, borderRadius: 99, padding: '2px 8px',
                  background: it.result === 'win' ? 'rgba(250,136,50,.16)' : 'rgba(10,10,11,.07)',
                  color: it.result === 'win' ? 'var(--orange-shade)' : 'var(--muted)',
                }}>{resultDE(it.result)}</span>
                <span className="tabular" style={{ fontSize: 12, color: 'var(--muted)' }}>{it.score}</span>
                {it.isNew && (
                  <span style={{ position: 'absolute', top: -7, right: 12, background: 'var(--orange)', color: 'var(--on-orange)', fontSize: 10.5, fontWeight: 700, borderRadius: 99, padding: '2px 8px', boxShadow: '0 2px 0 var(--orange-shade)' }}>Neu</span>
                )}
              </div>
            </div>
            <span style={{ color: 'var(--orange)', fontSize: 17 }}>›</span>
          </Link>
        ))}
      </div>
    </main>
  )
}
