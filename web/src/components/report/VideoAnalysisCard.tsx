import type { ReportPlayer } from '../../types'
import type { Anim } from '../../lib/motion'
import { useVideoPlayer } from '../../hooks/useVideoPlayer'
import { SectionCard } from './SectionCard'
import { VideoOverlay } from './VideoOverlay'
import { OverlayChips } from './OverlayChips'
import { VideoControls } from './VideoControls'

interface Props {
  slug: string | undefined
  video: { src: string; label: string }
  players: ReportPlayer[]
  anim?: Anim
}

/** Mobile "Video-Analyse" card: player surface + SVG overlay engine + toggle chips + controls.
 *  No real footage this slice → black backdrop with a synthetic demo clock (see useVideoPlayer). */
export function VideoAnalysisCard({ slug, video, players }: Props) {
  const hasFootage = Boolean(video.src)
  const player = useVideoPlayer(slug, hasFootage)

  return (
    <SectionCard
      title="Video-Analyse"
      sub={video.label}
      foot="Overlays aus der automatischen Auswertung — Demo-Daten auf Beispiel-Footage."
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
        <VideoOverlay time={player.time} players={players} ov={player.ov} />
      </div>

      <div style={{ marginTop: 12 }}>
        <OverlayChips ov={player.ov} toggle={player.toggle} />
      </div>
      <VideoControls player={player} style={{ marginTop: 12 }} />
    </SectionCard>
  )
}
