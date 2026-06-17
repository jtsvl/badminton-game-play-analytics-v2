import { useEffect, useRef, useState } from 'react'

/**
 * IntersectionObserver visibility with a 900 ms fallback so content in headless/clipped/never-
 * intersecting containers never stays stuck at opacity 0. Ported from the handoff `useInView`.
 */
export function useInView<T extends Element = HTMLElement>(threshold = 0.15) {
  const ref = useRef<T>(null)
  const [vis, setVis] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') {
      setVis(true)
      return
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) setVis(true) }),
      { threshold },
    )
    io.observe(el)
    const t = window.setTimeout(() => setVis(true), 900)
    return () => { io.disconnect(); window.clearTimeout(t) }
  }, [threshold])
  return [ref, vis] as const
}
