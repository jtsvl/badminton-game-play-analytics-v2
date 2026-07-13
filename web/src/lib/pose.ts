// Skeleton keyframe lookup over per-player pose_keyframes rows (specs/features/pose-skeleton-overlay.md).
// Keypoints are frame-normalized IMAGE coords (x,y ∈ [0,1] of the source frame), NOT court (u,v):
// elevated joints sit above the court plane where the ground homography is geometrically wrong.
// Deliberately nearest-frame, no interpolation — blending two real poses can produce anatomically
// impossible limbs; ~12 fps keyframes are dense enough to read as motion.

export interface PoseKey {
  t: number // seconds
  kp: [number, number, number][] // Halpe-26: [x, y, score] × 26
}

/**
 * Nearest keyframe to time t (keys sorted by t). An exact midpoint resolves to the later key.
 * Returns null when no key is within maxGapS (out-of-rally or track gap) so the skeleton hides
 * instead of freezing.
 */
export function poseAt(keys: PoseKey[], t: number, maxGapS = 0.3): PoseKey | null {
  if (keys.length === 0) return null
  // Binary search: lo = first index with keys[lo].t >= t (or the last index if t is past the end).
  let lo = 0
  let hi = keys.length - 1
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (keys[mid].t < t) lo = mid + 1
    else hi = mid
  }
  const nearest =
    lo > 0 && Math.abs(t - keys[lo - 1].t) < Math.abs(keys[lo].t - t) ? keys[lo - 1] : keys[lo]
  return Math.abs(nearest.t - t) <= maxGapS ? nearest : null
}
