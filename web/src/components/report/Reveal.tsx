import type { CSSProperties, ReactNode } from 'react'
import { useInView } from '../../hooks/useInView'
import { prefersReducedMotion, type Anim } from '../../lib/motion'

interface Props {
  i?: number
  anim?: Anim
  children: ReactNode
  style?: CSSProperties
}

/** Staggered entrance wrapper (opacity + translateY), `i` = stagger index. */
export function Reveal({ i = 0, anim = 'dezent', children, style }: Props) {
  const [ref, vis] = useInView<HTMLDivElement>(0.08)
  const on = prefersReducedMotion() || anim === 'aus' ? true : vis
  const dy = anim === 'voll' ? 20 : 10
  const step = anim === 'voll' ? 95 : 60
  return (
    <div
      ref={ref}
      style={{
        opacity: on ? 1 : 0,
        transform: on ? 'none' : `translateY(${dy}px)`,
        transition: `opacity .5s ease ${i * step}ms, transform .55s cubic-bezier(.22,.8,.3,1) ${i * step}ms`,
        ...style,
      }}
    >
      {children}
    </div>
  )
}
