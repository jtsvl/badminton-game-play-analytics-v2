import type { MatchResultCode } from '../../types'

interface Props {
  result: MatchResultCode | null
  line: string
  small?: boolean
}

/** Win/loss pill. Note our `result` is 'win'/'loss' (canonical), not the prototype's 'S'/'N'. */
export function ResultChip({ result, line, small }: Props) {
  const won = result === 'win'
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 5,
        padding: small ? '2px 8px' : '3px 10px',
        borderRadius: 99,
        fontSize: small ? 11 : 12,
        fontWeight: 700,
        whiteSpace: 'nowrap',
        background: won ? 'rgba(250,136,50,.16)' : 'rgba(10,10,11,.07)',
        color: won ? '#C66C27' : 'var(--muted)',
      }}
    >
      {line}
    </span>
  )
}
