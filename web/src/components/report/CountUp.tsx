import { useEffect, useState } from 'react'
import { deNum } from '../../lib/format'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'

interface Props {
  value: number
  decimals?: number
  anim?: Anim
  suffix?: string
}

/** Count up 0 → value (cubic ease-out, de-DE) on entry; always settles on the exact value. */
export function CountUp({ value, decimals = 0, anim = 'dezent', suffix = '' }: Props) {
  const [ref, vis] = useInView<HTMLSpanElement>(0.4)
  const dur = prefersReducedMotion() || anim === 'aus' ? 0 : anim === 'voll' ? 1200 : 700
  const [v, setV] = useState(dur ? 0 : value)

  useEffect(() => {
    if (!dur) {
      setV(value)
      return
    }
    if (!vis) return
    let raf = 0
    const t0 = performance.now()
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / dur)
      setV(value * (1 - Math.pow(1 - p, 3)))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    // rAF is paused in hidden tabs/iframes — always settle on the final value.
    const settle = window.setTimeout(() => setV(value), dur + 150)
    return () => {
      cancelAnimationFrame(raf)
      window.clearTimeout(settle)
    }
  }, [vis, value, dur])

  return (
    <span ref={ref}>
      {deNum(dur ? v : value, decimals)}
      {suffix}
    </span>
  )
}
