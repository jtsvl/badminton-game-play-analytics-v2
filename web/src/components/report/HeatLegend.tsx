/** wenig → viel gradient legend shown under a CourtHeatmap. */
export function HeatLegend() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 10 }}>
      <span style={{ fontSize: 11, color: 'var(--muted)' }}>wenig</span>
      <div
        style={{
          flex: 1,
          height: 6,
          borderRadius: 3,
          background: 'linear-gradient(90deg, rgba(250,136,50,.08), #FA8832 60%, #C0392B)',
        }}
      />
      <span style={{ fontSize: 11, color: 'var(--muted)' }}>viel</span>
    </div>
  )
}
