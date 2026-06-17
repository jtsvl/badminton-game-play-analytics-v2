// BC Rheinbach — Spielanalyse mock data (German, du-form)
// Shapes mirror web/src/types.ts of the analytics pipeline.
(function () {
  const singles = {
    video: { src: 'assets/match-clip.mp4', label: 'Demo-Aufnahme' },
    meta: {
      id: 1, dateLabel: '10. Juni 2026', weekday: 'Mittwoch', dd: '10.06.',
      discipline: 'Einzel', durationMin: 48, restMin: 14, court: 1,
      score: '21:18 · 19:21 · 21:15', games: [[21, 18], [19, 21], [21, 15]],
      result: 'S', resultLine: 'Sieg 2:1'
    },
    rallies: {
      count: 112, avgShots: 6.5, avgSec: 7.8, longShots: 34, longSec: 52,
      endLine: 'Der Smash beendete 31\u202f% aller Ballwechsel.'
    },
    pairs: null,
    players: [
      {
        name: 'Lukas Brandt', first: 'Lukas', isViewer: true, side: 'A',
        stats: {
          distance_m: 2840, avg_kmh: 5.9, top_kmh: 21.4, sprints: 27,
          zones: { front: 31, mid: 44, rear: 25, left: 48, right: 52 },
          timeline: [225, 300, 330, 275, 290, 340, 250, 280, 310, 240],
          heat: [
            { x: 0.50, y: 0.55, w: 1.0, s: 0.18 },
            { x: 0.32, y: 0.28, w: 0.55, s: 0.13 },
            { x: 0.70, y: 0.80, w: 0.60, s: 0.15 },
            { x: 0.74, y: 0.30, w: 0.40, s: 0.12 }
          ],
          shots: [['Clear', 84], ['Netzdrop', 62], ['Drop', 56], ['Lift', 47], ['Drive', 41], ['Aufschlag', 39], ['Smash', 38]],
          fastest: 196
        }
      },
      {
        name: 'Jonas Weber', first: 'Jonas', isViewer: false, side: 'B',
        stats: {
          distance_m: 2610, avg_kmh: 5.4, top_kmh: 19.8, sprints: 21,
          zones: { front: 26, mid: 41, rear: 33, left: 51, right: 49 },
          timeline: [200, 270, 300, 255, 265, 310, 235, 260, 290, 225],
          heat: [
            { x: 0.48, y: 0.68, w: 1.0, s: 0.18 },
            { x: 0.28, y: 0.45, w: 0.50, s: 0.13 },
            { x: 0.72, y: 0.50, w: 0.55, s: 0.14 },
            { x: 0.52, y: 0.28, w: 0.30, s: 0.11 }
          ],
          shots: [['Clear', 91], ['Lift', 58], ['Netzdrop', 55], ['Drop', 49], ['Aufschlag', 38], ['Drive', 37], ['Smash', 29]],
          fastest: 178
        }
      }
    ]
  };

  const doubles = {
    meta: {
      id: 2, dateLabel: '3. Juni 2026', weekday: 'Mittwoch', dd: '03.06.',
      discipline: 'Doppel', durationMin: 39, restMin: 11, court: 2,
      score: '21:17 · 21:19', games: [[21, 17], [21, 19]],
      result: 'S', resultLine: 'Sieg 2:0'
    },
    rallies: {
      count: 89, avgShots: 7.1, avgSec: 8.4, longShots: 41, longSec: 58,
      endLine: 'Der Smash beendete 36\u202f% aller Ballwechsel.'
    },
    pairs: [
      { side: 'A', label: 'Lukas & Miriam', fb: 58, ss: 42, rot: 46 },
      { side: 'B', label: 'David & Anna', fb: 41, ss: 59, rot: 31 }
    ],
    players: [
      {
        name: 'Lukas Brandt', first: 'Lukas', isViewer: true, side: 'A',
        stats: {
          distance_m: 1980, avg_kmh: 5.6, top_kmh: 20.2, sprints: 18,
          zones: { front: 44, mid: 38, rear: 18, left: 46, right: 54 },
          timeline: [230, 260, 275, 240, 255, 270, 245, 205],
          heat: [
            { x: 0.50, y: 0.32, w: 1.0, s: 0.16 },
            { x: 0.30, y: 0.50, w: 0.50, s: 0.13 },
            { x: 0.68, y: 0.55, w: 0.45, s: 0.13 },
            { x: 0.50, y: 0.75, w: 0.25, s: 0.12 }
          ],
          shots: [['Netzdrop', 41], ['Drive', 38], ['Smash', 34], ['Clear', 22], ['Aufschlag', 21], ['Drop', 18], ['Lift', 12]],
          fastest: 189
        }
      },
      {
        name: 'Miriam Schulte', first: 'Miriam', isViewer: false, side: 'A',
        stats: {
          distance_m: 1840, avg_kmh: 5.2, top_kmh: 18.9, sprints: 14,
          zones: { front: 22, mid: 39, rear: 39, left: 52, right: 48 },
          timeline: [215, 235, 250, 225, 240, 255, 230, 190],
          heat: [
            { x: 0.45, y: 0.72, w: 1.0, s: 0.17 },
            { x: 0.70, y: 0.60, w: 0.50, s: 0.13 },
            { x: 0.30, y: 0.62, w: 0.45, s: 0.13 },
            { x: 0.50, y: 0.35, w: 0.20, s: 0.11 }
          ],
          shots: [['Clear', 48], ['Lift', 35], ['Smash', 28], ['Drop', 26], ['Drive', 24], ['Aufschlag', 20], ['Netzdrop', 14]],
          fastest: 172
        }
      },
      {
        name: 'David Krämer', first: 'David', isViewer: false, side: 'B',
        stats: {
          distance_m: 1905, avg_kmh: 5.5, top_kmh: 19.6, sprints: 17,
          zones: { front: 38, mid: 40, rear: 22, left: 49, right: 51 },
          timeline: [225, 245, 260, 235, 245, 260, 235, 200],
          heat: [
            { x: 0.52, y: 0.40, w: 1.0, s: 0.16 },
            { x: 0.28, y: 0.55, w: 0.55, s: 0.13 },
            { x: 0.72, y: 0.62, w: 0.40, s: 0.13 },
            { x: 0.50, y: 0.78, w: 0.25, s: 0.11 }
          ],
          shots: [['Clear', 41], ['Drive', 33], ['Lift', 30], ['Smash', 25], ['Drop', 21], ['Aufschlag', 20], ['Netzdrop', 19]],
          fastest: 181
        }
      },
      {
        name: 'Anna Lindner', first: 'Anna', isViewer: false, side: 'B',
        stats: {
          distance_m: 1760, avg_kmh: 4.9, top_kmh: 17.8, sprints: 12,
          zones: { front: 25, mid: 41, rear: 34, left: 50, right: 50 },
          timeline: [205, 220, 235, 210, 225, 235, 225, 205],
          heat: [
            { x: 0.50, y: 0.68, w: 1.0, s: 0.17 },
            { x: 0.68, y: 0.50, w: 0.45, s: 0.13 },
            { x: 0.32, y: 0.58, w: 0.50, s: 0.13 },
            { x: 0.48, y: 0.32, w: 0.20, s: 0.11 }
          ],
          shots: [['Netzdrop', 36], ['Drive', 30], ['Clear', 26], ['Drop', 22], ['Smash', 21], ['Aufschlag', 21], ['Lift', 17]],
          fastest: 168
        }
      }
    ]
  };

  // Build a variant report from a base (other matches in the list).
  function variant(base, meta, rallies, names, factor) {
    const r = JSON.parse(JSON.stringify(base));
    r.meta = meta;
    if (rallies) r.rallies = Object.assign(r.rallies, rallies);
    r.players.forEach(function (p, i) {
      if (names[i]) {
        p.name = names[i];
        p.first = names[i].split(' ')[0];
        p.isViewer = names[i] === 'Lukas Brandt';
      }
      const s = p.stats;
      s.distance_m = Math.round(s.distance_m * factor / 10) * 10;
      s.sprints = Math.round(s.sprints * factor);
      s.avg_kmh = Math.round(s.avg_kmh * (0.9 + 0.2 * factor) * 10) / 10;
      s.top_kmh = Math.round(s.top_kmh * (0.92 + 0.12 * factor) * 10) / 10;
      s.timeline = s.timeline.map(function (v) { return Math.round(v * factor); });
      s.shots = s.shots.map(function (sh) { return [sh[0], Math.max(4, Math.round(sh[1] * factor))]; });
      if (s.fastest) s.fastest = Math.round(s.fastest * (0.94 + 0.1 * factor));
    });
    if (r.pairs && names.length === 4) {
      r.pairs[0].label = r.players[0].first + ' & ' + r.players[1].first;
      r.pairs[1].label = r.players[2].first + ' & ' + r.players[3].first;
    }
    return r;
  }

  const reports = {
    1: singles,
    2: doubles,
    3: variant(singles,
      { id: 3, dateLabel: '27. Mai 2026', weekday: 'Mittwoch', dd: '27.05.', discipline: 'Einzel', durationMin: 41, restMin: 12, court: 1, score: '18:21 · 21:23', games: [[18, 21], [21, 23]], result: 'N', resultLine: 'Niederlage 0:2' },
      { count: 96, avgShots: 6.1, avgSec: 7.2, longShots: 28, longSec: 44, endLine: 'Der Smash beendete 27\u202f% aller Ballwechsel.' },
      ['Lukas Brandt', 'Tobias Vogt'], 0.9),
    4: variant(doubles,
      { id: 4, dateLabel: '20. Mai 2026', weekday: 'Mittwoch', dd: '20.05.', discipline: 'Doppel', durationMin: 52, restMin: 15, court: 2, score: '21:12 · 19:21 · 21:18', games: [[21, 12], [19, 21], [21, 18]], result: 'S', resultLine: 'Sieg 2:1' },
      { count: 118, avgShots: 7.4, avgSec: 8.8, longShots: 45, longSec: 63, endLine: 'Der Smash beendete 34\u202f% aller Ballwechsel.' },
      ['Lukas Brandt', 'Markus Hoffmann', 'Elif Kaya', 'Jonas Weber'], 1.07),
    5: variant(singles,
      { id: 5, dateLabel: '13. Mai 2026', weekday: 'Mittwoch', dd: '13.05.', discipline: 'Einzel', durationMin: 33, restMin: 9, court: 1, score: '21:9 · 21:14', games: [[21, 9], [21, 14]], result: 'S', resultLine: 'Sieg 2:0' },
      { count: 78, avgShots: 5.6, avgSec: 6.5, longShots: 24, longSec: 38, endLine: 'Der Netzdrop beendete 29\u202f% aller Ballwechsel.' },
      ['Lukas Brandt', 'David Krämer'], 0.78),
  };

  const matches = [
    { id: 1, dd: '10.06.', day: '10', month: 'Juni', dateLabel: '10. Juni 2026', discipline: 'Einzel', a: 'Lukas Brandt', b: 'Jonas Weber', score: '21:18 · 19:21 · 21:15', result: 'S', isNew: true },
    { id: 2, dd: '03.06.', day: '3', month: 'Juni', dateLabel: '3. Juni 2026', discipline: 'Doppel', a: 'Lukas Brandt / Miriam Schulte', b: 'David Krämer / Anna Lindner', score: '21:17 · 21:19', result: 'S', isNew: false },
    { id: 3, dd: '27.05.', day: '27', month: 'Mai', dateLabel: '27. Mai 2026', discipline: 'Einzel', a: 'Lukas Brandt', b: 'Tobias Vogt', score: '18:21 · 21:23', result: 'N', isNew: false },
    { id: 4, dd: '20.05.', day: '20', month: 'Mai', dateLabel: '20. Mai 2026', discipline: 'Doppel', a: 'Lukas Brandt / Markus Hoffmann', b: 'Elif Kaya / Jonas Weber', score: '21:12 · 19:21 · 21:18', result: 'S', isNew: false },
    { id: 5, dd: '13.05.', day: '13', month: 'Mai', dateLabel: '13. Mai 2026', discipline: 'Einzel', a: 'Lukas Brandt', b: 'David Krämer', score: '21:9 · 21:14', result: 'S', isNew: false }
  ];

  window.BCA = { viewer: 'Lukas Brandt', viewerFirst: 'Lukas', matches: matches, reports: reports };
})();
