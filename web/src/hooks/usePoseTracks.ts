import { useEffect, useRef, useState } from 'react'
import { supabase } from '../lib/supabase'
import type { PoseKey } from '../lib/pose'
import type { PoseKeyframeRow } from '../types'

const EMPTY = new Map<string, PoseKey[]>()

/** Lazy pose_keyframes fetch: skeleton payloads are hundreds of KB per match, so nothing is
 *  downloaded until the Skelett chip is first toggled on; the result is kept for the session
 *  (toggling off and on again does not refetch). fetchedFor is a ref, not state — as a dep it
 *  would re-run the effect and a cleanup-side cancel flag would discard the in-flight response. */
export function usePoseTracks(videoId: string | null | undefined, enabled: boolean): Map<string, PoseKey[]> {
  const [tracks, setTracks] = useState<Map<string, PoseKey[]>>(EMPTY)
  const fetchedFor = useRef<string | null>(null)

  useEffect(() => {
    if (!enabled || !videoId || fetchedFor.current === videoId) return
    fetchedFor.current = videoId
    supabase
      .from('pose_keyframes')
      .select('subject, data')
      .eq('video_id', videoId)
      .then(({ data }) => {
        const rows = (data ?? []) as PoseKeyframeRow[]
        setTracks(new Map(rows.map((r) => [r.subject, r.data])))
      })
  }, [enabled, videoId])

  return tracks
}
