// Resolve Supabase Storage paths to signed URLs. Buckets are private, so the anon client needs a
// storage read policy (dev: supabase/migrations/0003_dev_calibration_write.sql; prod: player-link RLS).
// Awaited inside useMatchReport's single fetch so the report arrives with a ready src.
import { supabase } from './supabase'

/** Signed URL for an object in a bucket (1 h TTL), or '' if unavailable. */
export async function signedUrl(bucket: string, path: string | null, ttl = 60 * 60): Promise<string> {
  if (!path) return ''
  try {
    const { data } = await supabase.storage.from(bucket).createSignedUrl(path, ttl)
    return data?.signedUrl ?? ''
  } catch {
    return ''
  }
}

/** Playable URL for a match video storage path ('' → no source → synthetic demo clock fallback). */
export function resolveVideoSrc(storagePath: string | null): Promise<string> {
  return signedUrl('match-videos', storagePath)
}
