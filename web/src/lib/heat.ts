// Rasterise gaussian heat kernels into a normalised cols×rows grid for CourtHeatmap.
// Kernel shape (specs/20-data-model.md): { x, y, w, s } with x,y∈[0,1] (left→right, far→near),
// w = weight, s = sigma. Ported from the handoff `heatGrid`.

export interface HeatSpot { x: number; y: number; w: number; s: number }
export interface HeatCell { c: number; r: number; v: number }

export function heatGrid(spots: HeatSpot[], cols: number, rows: number): HeatCell[] {
  const g: HeatCell[] = []
  let max = 0
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = (c + 0.5) / cols
      const y = (r + 0.5) / rows
      let v = 0
      for (const s of spots) {
        const dx = x - s.x
        const dy = y - s.y
        v += s.w * Math.exp(-(dx * dx + dy * dy) / (2 * s.s * s.s))
      }
      g.push({ c, r, v })
      if (v > max) max = v
    }
  }
  if (max > 0) for (const cell of g) cell.v /= max
  return g
}
