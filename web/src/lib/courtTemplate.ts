// BWF court line segments in court coordinates (u,v): u left->right, v far(0)->near(1) baseline.
// Pure data — projected through the live homography to draw a registration grid over the
// calibration still (so the trainer drags corners until the orange grid sits on the painted lines).
// Distances are normalised over the full doubles court: 6.10 m wide x 13.40 m long.

export type Seg = readonly [readonly [number, number], readonly [number, number]]

export const NET_V = 0.5
// Singles sidelines sit (6.10-5.18)/2 = 0.46 m inside each doubles sideline.
export const SINGLES_X = 0.46 / 6.1 // ~0.0754
// Short service line is 1.98 m from the net; doubles long service line 0.76 m from each baseline.
export const SHORT_SERVICE_DV = 1.98 / 13.4 // ~0.1478
export const LONG_SERVICE_DV = 0.76 / 13.4 // ~0.0567

const FRONT_SHORT = NET_V - SHORT_SERVICE_DV // far-side short service line
const NEAR_SHORT = NET_V + SHORT_SERVICE_DV // near-side short service line

export const COURT_TEMPLATE: Seg[] = [
  // outer doubles boundary (closed quad)
  [[0, 0], [1, 0]],
  [[1, 0], [1, 1]],
  [[1, 1], [0, 1]],
  [[0, 1], [0, 0]],
  // singles sidelines (full length)
  [[SINGLES_X, 0], [SINGLES_X, 1]],
  [[1 - SINGLES_X, 0], [1 - SINGLES_X, 1]],
  // net
  [[0, NET_V], [1, NET_V]],
  // short service lines
  [[0, FRONT_SHORT], [1, FRONT_SHORT]],
  [[0, NEAR_SHORT], [1, NEAR_SHORT]],
  // doubles long (back) service lines
  [[0, LONG_SERVICE_DV], [1, LONG_SERVICE_DV]],
  [[0, 1 - LONG_SERVICE_DV], [1, 1 - LONG_SERVICE_DV]],
  // centre service line on each half (short service line -> baseline)
  [[0.5, 0], [0.5, FRONT_SHORT]],
  [[0.5, NEAR_SHORT], [0.5, 1]],
]
