import { Link } from 'react-router-dom'

// P0 STUB. Full upload + calibration flow is specs/10 FR-1..FR-8 (P0/P1).
export function Upload() {
  return (
    <main style={{ maxWidth: 480, margin: '0 auto', padding: 24 }}>
      <Link to="/" style={{ fontSize: 13, color: 'var(--muted)' }}>‹ Zurück</Link>
      <h1 style={{ fontSize: 27, fontWeight: 700, marginTop: 12 }}>Aufnahme hochladen</h1>
      <p style={{ color: 'var(--muted)', fontSize: 14 }}>
        Upload, Spielanlage und Feld-Kalibrierung folgen (FR-1–FR-8).
      </p>
    </main>
  )
}
