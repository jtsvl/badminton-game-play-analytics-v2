import { useCallback, useEffect, useRef, useState } from 'react'
import { DEMO_LOOP } from '../lib/demoOverlay'

export interface OverlayToggles {
  tracking: boolean
  wege: boolean
  shuttle: boolean
  zonen: boolean
}

export interface VideoPlayer {
  vref: React.RefObject<HTMLVideoElement>
  playing: boolean
  setPlaying: (b: boolean) => void
  time: number
  setTime: (t: number) => void
  dur: number
  hasFootage: boolean
  ov: OverlayToggles
  toggle: (key: keyof OverlayToggles) => void
  onLoaded: () => void
  togglePlay: () => void
  seek: (val: number) => void
  rate: number
  setRate: (r: number) => void
}

const OV_KEY = 'bca-ov-v1'
const RATE_KEY = 'bca-rate-v1'
const DEFAULT_OV: OverlayToggles = { tracking: true, wege: true, shuttle: false, zonen: false }

function loadRate(): number {
  try {
    const r = parseFloat(localStorage.getItem(RATE_KEY) || '')
    if (r >= 0.1 && r <= 4) return r
  } catch {
    /* noop */
  }
  return 1
}

function loadOv(): OverlayToggles {
  try {
    const raw = localStorage.getItem(OV_KEY)
    if (raw) {
      const s = JSON.parse(raw)
      if (s && typeof s === 'object') return { ...DEFAULT_OV, ...s }
    }
  } catch {
    /* noop */
  }
  return DEFAULT_OV
}

/**
 * Video playback + overlay state. When there is no playable footage (this slice), a synthetic clock
 * advances `time` over the demo loop so the overlay still animates over a black backdrop.
 * Position persists in localStorage['bca-vt-'+slug]; overlay toggles in localStorage['bca-ov-v1'].
 */
export function useVideoPlayer(slug: string | undefined, hasFootage: boolean): VideoPlayer {
  const vref = useRef<HTMLVideoElement>(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [dur, setDur] = useState(hasFootage ? 0 : DEMO_LOOP)
  const [ov, setOv] = useState<OverlayToggles>(loadOv)
  const [rate, setRateState] = useState<number>(loadRate)
  const posKey = 'bca-vt-' + (slug ?? 'x')

  const toggle = useCallback((key: keyof OverlayToggles) => {
    setOv((prev) => {
      const next = { ...prev, [key]: !prev[key] }
      try {
        localStorage.setItem(OV_KEY, JSON.stringify(next))
      } catch {
        /* noop */
      }
      return next
    })
  }, [])

  // Playback clock: real footage drives from the <video>; otherwise a synthetic loop.
  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const v = vref.current
      if (hasFootage && v) {
        setTime(v.currentTime)
      } else {
        const dt = (now - last) / 1000
        last = now
        setTime((t) => {
          const n = t + dt
          return n >= DEMO_LOOP ? n - DEMO_LOOP : n
        })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, hasFootage])

  // Persist playback position at 1 s granularity (real footage only — demo restarts each mount).
  const sec = Math.floor(time)
  useEffect(() => {
    if (hasFootage && sec > 0) {
      try {
        localStorage.setItem(posKey, String(sec))
      } catch {
        /* noop */
      }
    }
  }, [sec, hasFootage, posKey])

  const onLoaded = useCallback(() => {
    const v = vref.current
    if (!v) return
    setDur(v.duration || 0)
    v.playbackRate = rate // a freshly-loaded <video> resets to 1× — re-apply the chosen speed
    try {
      const saved = parseFloat(localStorage.getItem(posKey) || '')
      if (saved > 0 && saved < (v.duration || 0) - 1) {
        v.currentTime = saved
        setTime(saved)
      } else {
        v.currentTime = 1
      }
    } catch {
      v.currentTime = 1
    }
  }, [posKey, rate])

  const togglePlay = useCallback(() => {
    const v = vref.current
    if (hasFootage && v) {
      if (v.paused) {
        void v.play()
        setPlaying(true)
      } else {
        v.pause()
        setPlaying(false)
      }
    } else {
      setPlaying((p) => !p)
    }
  }, [hasFootage])

  const seek = useCallback(
    (val: number) => {
      const v = vref.current
      if (hasFootage && v) {
        if (!v.duration) return
        v.currentTime = val
      }
      setTime(val)
    },
    [hasFootage],
  )

  const setRate = useCallback((r: number) => {
    const v = vref.current
    if (v) v.playbackRate = r
    setRateState(r)
    try {
      localStorage.setItem(RATE_KEY, String(r))
    } catch {
      /* noop */
    }
  }, [])

  return { vref, playing, setPlaying, time, setTime, dur, hasFootage, ov, toggle, onLoaded, togglePlay, seek, rate, setRate }
}
