// Overlay track interpolation over downsampled court-coordinate keyframes.
// Canonical keyframe shape (specs/20-data-model.md): [{ t, u, v }] with t∈[0,1] along a loop,
// u (left→right) and v (far baseline→near baseline) ∈ [0,1]. In P1 these come from
// `track_keyframes`; for the demo from lib/demoOverlay.ts.

export interface TrackKey { t: number; u: number; v: number }
export interface UV { u: number; v: number }

/** Position along a keyframe track at normalised time p∈[0,1] (smoothstep between keys, clamped at ends). */
export function trackAt(keys: TrackKey[], p: number): UV {
  if (keys.length === 0) return { u: 0.5, v: 0.5 }
  if (p <= keys[0].t) return { u: keys[0].u, v: keys[0].v }
  for (let i = 0; i < keys.length - 1; i++) {
    const a = keys[i]
    const b = keys[i + 1]
    if (p >= a.t && p <= b.t) {
      const raw = (p - a.t) / (b.t - a.t)
      const s = raw * raw * (3 - 2 * raw) // smoothstep
      return { u: a.u + (b.u - a.u) * s, v: a.v + (b.v - a.v) * s }
    }
  }
  const last = keys[keys.length - 1]
  return { u: last.u, v: last.v }
}

export interface ShuttleState extends UV {
  segP: number
  from: UV
}

/**
 * Shuttle position for a demo rally: the loop is split into `segments` hits that alternate
 * near→far→near…, each interpolated linearly between the two players' positions at the segment ends.
 */
export function shuttleAt(
  p: number,
  near: TrackKey[],
  far: TrackKey[],
  segments = 10,
): ShuttleState {
  const seg = Math.floor(p * segments) % segments
  const segP = (p * segments) % 1
  const tA = seg / segments
  const tBraw = (seg + 1) / segments
  const fromNear = seg % 2 === 0
  const a = trackAt(fromNear ? near : far, tA)
  const b = trackAt(fromNear ? far : near, tBraw > 1 ? tBraw - 1 : tBraw)
  return {
    u: a.u + (b.u - a.u) * segP,
    v: a.v + (b.v - a.v) * segP,
    segP,
    from: { u: a.u, v: a.v },
  }
}
