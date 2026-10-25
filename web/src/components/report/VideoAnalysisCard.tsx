import type { ReportPlayer } from '../../types'
import type { Anim } from '../../lib/motion'
import type { OverlayInputs } from '../../lib/calibration'
import { useVideoPlayer } from '../../hooks/useVideoPlayer'
import { usePoseTracks } from '../../hooks/usePoseTracks'
import { SectionCard } from './SectionCard'
import { VideoOverlay } from './VideoOverlay'
import { SkeletonOverlay } from './SkeletonOverlay'
import { OverlayChips } from './OverlayChips'
import { VideoControls } from './VideoControls'

interface Props {
  slug: string | undefined
  video: { id?: string; src: string; label: string; durationS?: number | null }
  players: ReportPlayer[]
  oi: OverlayInputs
  anim?: Anim
}

/** Mobile "Video-Analyse" card: player surface + SVG overlay engine + toggle chips + controls.
 *  Real footage → signed-URL video + overlays from the computed track_keyframes; otherwise the demo. */
export function VideoAnalysisCard({ slug, video, players, oi }: Props) {
  const hasFootage = Boolean(video.src)
  const player = useVideoPlayer(slug, hasFootage)
  const poseTracks = usePoseTracks(video.id, player.ov.skelett)

  return (
    <SectionCard
      title="Video-Analyse"
      sub={video.label}
      foot={
        oi.isDemo
          ? 'Overlays aus der automatischen Auswertung — Demo-Daten auf Beispiel-Footage.'
          : 'Overlays aus der automatischen Auswertung dieser Aufnahme.'
      }
    >
      <div
        style={{
          position: 'relative',
          borderRadius: 10,
          overflow: 'hidden',
          background: '#0A0A0B',
          lineHeight: 0,
          aspectRatio: hasFootage ? undefined : '16 / 9',
        }}
      >
        {hasFootage && (
          <video
            ref={player.vref}
            src={video.src}
            muted
            playsInline
            preload="metadata"
            onLoadedMetadata={player.onLoaded}
            onEnded={() => player.setPlaying(false)}
            onTimeUpdate={() => {
              const v = player.vref.current
              if (v && v.paused) player.setTime(v.currentTime)
            }}
            style={{ width: '100%', display: 'block' }}
          />
        )}
        <VideoOverlay time={player.time} players={players} ov={player.ov} H={oi.H} tracks={oi.tracks} />
        <SkeletonOverlay time={player.time} players={players} ov={player.ov} poseTracks={poseTracks} />
      </div>

      <div style={{ marginTop: 12 }}>
        <OverlayChips ov={player.ov} toggle={player.toggle} />
      </div>
      <VideoControls player={player} style={{ marginTop: 12 }} />
    </SectionCard>
  )
}
