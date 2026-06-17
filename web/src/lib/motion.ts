// Animation level + reduced-motion guard. The handoff exposes three levels; we ship 'dezent'
// (subtle) as the default. Components also short-circuit in JS when the user prefers reduced motion,
// since the global `transition: none` CSS alone would otherwise leave them at their start state.
export type Anim = 'aus' | 'dezent' | 'voll'

export function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  )
}
