// Resolve a Supabase Storage path to a playable video URL.
//
// This slice: the demo clip is not in Storage (all buckets are private and the referenced object is
// absent), so there is no playable source — the video surface falls back to a synthetic demo clock
// over a black backdrop while still rendering the SVG overlay engine. We return '' to signal "no source".
//
// TODO (P1): once real footage is uploaded, return a signed URL, e.g.
//   const { data } = await supabase.storage.from('match-videos').createSignedUrl(path, 60 * 60)
//   return data?.signedUrl ?? ''
// (async — resolve inside useMatchReport's single fetch so the report arrives with a ready src).
export function resolveVideoSrc(_storagePath: string | null): string {
  return ''
}
