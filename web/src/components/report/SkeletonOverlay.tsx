import type { ReportPlayer } from '../../types'
import { poseAt, type PoseKey } from '../../lib/pose'
import type { OverlayToggles } from '../../hooks/useVideoPlayer'

interface Props {
  time: number // seconds (real video.currentTime, or the synthetic demo clock)
  players: ReportPlayer[]
  ov: OverlayToggles
  poseTracks: Map<string, PoseKey[]> // by subject 'player:<id>'; t in seconds
}

const VB_W = 192
const VB_H = 108
const ORANGE = '#FA8832'
const BLUE = '#1758A6'
const MIN_SCORE = 0.3 // hide joints/bones below this RTMPose confidence

// Halpe-26 (verified): 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees,
// 15/16 ankles, 17 head, 18 neck, 19 pelvis, 20/21 big toes, 22/23 small toes, 24/25 heels.
// Face points 0-4 exist in the data but are deliberately not drawn.
const BONES: [number, number][] = [
  [18, 17], [18, 5], [18, 6],
  [5, 7], [7, 9], [6, 8], [8, 10],
  [18, 19], [19, 11], [19, 12],
  [11, 13], [13, 15], [15, 20], [15, 22], [15, 24],
  [12, 14], [14, 16], [16, 21], [16, 23], [16, 25],
]

// Fade weak detections instead of popping bones in and out at the MIN_SCORE cliff.
const fade = (score: number) => Math.min(0.9, Math.max(0.35, score))

/**
 * Per-player skeleton over the video surface, sibling of VideoOverlay. CRITICAL difference: pose
 * keypoints are frame-normalized IMAGE coords (x,y ∈ [0,1] of the source frame), NOT court (u,v) —
 * rendered directly as x*VB_W / y*VB_H with no homography (elevated joints sit above the court
 * plane where the ground homography is geometrically wrong). Nearest-frame lookup via poseAt;
 * beyond its max gap (out-of-rally / track gap) the skeleton hides instead of freezing.
 */
export function SkeletonOverlay({ time, players, ov, poseTracks }: Props) {
  if (!ov.skelett) return null

  const viewerSide = players[0]?.side
  const colorFor = (p: ReportPlayer) => (p.side === viewerSide ? ORANGE : BLUE)

  const skeleton = (pose: PoseKey, color: string) => (
    <g stroke={color} strokeLinecap="round">
      {BONES.map(([a, b], i) => {
        const ka = pose.kp[a]
        const kb = pose.kp[b]
        if (!ka || !kb) return null
        const score = Math.min(ka[2], kb[2])
        if (score < MIN_SCORE) return null
        return (
          <line
            key={i}
            x1={ka[0] * VB_W}
            y1={ka[1] * VB_H}
            x2={kb[0] * VB_W}
            y2={kb[1] * VB_H}
            strokeWidth="0.6"
            opacity={fade(score)}
          />
        )
      })}
      {pose.kp.map(([x, y, score], i) =>
        i < 5 || score < MIN_SCORE ? null : (
          <circle key={i} cx={x * VB_W} cy={y * VB_H} r="0.7" fill={color} stroke="none" opacity={fade(score)} />
        ),
      )}
    </g>
  )

  const posed = players
    .map((p) => ({ p, pose: poseAt(poseTracks.get(`player:${p.playerId}`) ?? [], time) }))
    .filter((x): x is { p: ReportPlayer; pose: PoseKey } => x.pose !== null)

  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      preserveAspectRatio="none"
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
    >
      {posed.map(({ p, pose }) => (
        <g key={p.playerId}>{skeleton(pose, colorFor(p))}</g>
      ))}
    </svg>
  )
}
