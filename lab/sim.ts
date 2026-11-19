// Lab: an agent-based simulator of a big launch's Exophase trophy page, to calibrate main.ts offline.
// It is deliberately richer than the forecaster's own model (diurnal play, quitting, correlated player speed,
// trophies earned in sessions, irregular refreshes), so the scores here are an honest-ish check.
//
//   node lab/sim.ts [worlds=40] [seed=1]
// prints the mean logit pinball loss and 90% coverage of main.ts and of the two house baselines.

import { forecast, logitPct, P, type Case, type Read, type Trophy } from '../main.ts';

export function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const gauss = (r: () => number): number => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const logU = (r: () => number, a: number, b: number): number => Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * r());

const T0 = Date.parse('2026-11-19T00:00:00Z');
const iso = (h: number): string => new Date(T0 + h * 3_600_000).toISOString().replace('.000Z', 'Z');

export interface World { c: Case; answer: number[] }

/** One synthetic world: a final-shaped case (cut at tc hours, target at tT) and its answer rates. */
export function makeWorld(seed: number, tc = 59.5, tT = 168): World {
  const r = rng(seed);
  const players = 12000; // simulated; the page's count is scaled up
  const scale = logU(r, 8, 60);
  const tau = logU(r, 20, 160); // arrival decay, hours
  const diurnalA = 0.6 * r();
  const phase = 24 * r();
  const speedMed = logU(r, 1.5, 5) / 24; // play hours per wall-clock hour
  const quitMean = logU(r, 60, 600);
  // arrival times by rejection from exp decay * diurnal, starting ~4h after T0
  const start = 2 + 4 * r();
  const arrive: number[] = [];
  while (arrive.length < players) {
    const t = start + -tau * Math.log(1 - r() * (1 - Math.exp(-(400 - start) / tau)));
    const d = 1 + diurnalA * Math.sin((2 * Math.PI * (t - phase)) / 24);
    if (r() * (1 + diurnalA) < d) arrive.push(t);
  }
  arrive.sort((a, b) => a - b);
  const speed = arrive.map(() => speedMed * Math.exp(0.7 * gauss(r)));
  const quit = arrive.map(() => (r() < 0.3 ? Infinity : -quitMean * Math.log(1 - r())));
  const skill = arrive.map(() => gauss(r));
  // trophies: first auto trophy, a story chain, side trophies, completionist
  const trophies: Trophy[] = [];
  const reqH: number[] = []; const att: number[] = [];
  const add = (name: string, tier: string, h: number, p: number, secret: boolean): void => {
    trophies.push({ id: trophies.length + 1, name, description: name, tier, points: tier === 'bronze' ? 15 : tier === 'silver' ? 30 : tier === 'gold' ? 90 : 300, secret });
    reqH.push(h); att.push(p);
  };
  const storyScale = logU(r, 0.5, 2);
  add('Platinum', 'platinum', 250 * storyScale, 0.3, false);
  add('First', 'bronze', 0.05, 1, true);
  const chain = [5, 11, 18, 26, 55];
  chain.forEach((h, i) => add(`Chapter ${i + 1}`, i === 4 ? 'gold' : 'bronze', h * storyScale, 0.97 - 0.05 * i, true));
  for (let i = 0; i < 26; i++) add(`Side ${i}`, 'bronze', logU(r, 0.3, 120) * storyScale, logU(r, 0.03, 0.95), r() < 0.4);
  for (let i = 0; i < 12; i++) add(`Grind ${i}`, 'silver', logU(r, 40, 300) * storyScale, logU(r, 0.05, 0.6), false);
  add('Completion', 'gold', 230 * storyScale, 0.3, false);
  // the real page's wording for the story chain and the completionist trophy
  for (const t of trophies) {
    const m = /^Chapter (\d)$/.exec(t.name);
    if (m) t.description = m[1] === '5' ? 'Complete the final mission.' : `Complete Chapter ${m[1]}.`;
    if (t.name === 'Completion') t.description = 'Attain 100% Completion.';
  }
  // earn times
  const earn: number[][] = trophies.map((_, k) => {
    const ts: number[] = [];
    for (let i = 0; i < players; i++) {
      if (r() > att[k]) continue;
      const need = reqH[k] * Math.exp(0.35 * skill[i] + 0.25 * gauss(r));
      const wall = k === 1 ? 0 : need / speed[i]; // Exophase counts a player once they hold a trophy: the first one
      if (wall > quit[i]) continue;
      // play happens in the evening: snap to a session boundary
      const t = arrive[i] + wall * (1 + 0.15 * Math.sin((2 * Math.PI * (arrive[i] + wall - phase)) / 24));
      ts.push(t);
    }
    return ts.sort((a, b) => a - b);
  });
  const countBefore = (arr: number[], t: number): number => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= t) lo = m + 1; else hi = m; } return lo; };
  // launch-night cohort: 638 players, a few far along
  const base = 638;
  const baseRates = trophies.map((t, k) => (k === 1 ? 100 : round2((100 * Math.round(base * Math.min(0.02, att[k] * 0.01 * Math.exp(-reqH[k] / 80) + 0.006))) / base)));
  const baseCount = baseRates.map((x) => (x * base) / 100);
  const page = (t: number): { n: number; rates: number[] } => {
    const n = countBefore(arrive, t);
    const N = base + n * scale;
    return { n: Math.round(N), rates: trophies.map((_, k) => round2(Math.min(100, (100 * (baseCount[k] + countBefore(earn[k], t) * scale)) / N))) };
  };
  // refresh clock: irregular and slow (live page sat unchanged 14:00..18:30 UTC on launch day), first one at 4.5h
  const refreshes: number[] = [];
  const gapMean = process.env.FASTREFRESH ? logU(r, 0.4, 3) : logU(r, 2, 20);
  for (let t = 4.5; t < tT + 2; t += gapMean * (0.3 + 1.4 * r())) refreshes.push(t);
  const history: Read[] = [];
  const read = (slotH: number): { n: number; rates: number[] } => {
    let last = -Infinity;
    for (const f of refreshes) if (f <= slotH) last = f;
    return Number.isFinite(last) ? page(last) : { n: base, rates: baseRates };
  };
  for (let h = -10 / 60; h < tc; h += h < 40 ? 0.5 : 1 / 6) {
    const pg = read(h);
    history.push({ slot: iso(h), at: iso(h + 0.0001), players_tracked: pg.n, rates: pg.rates });
  }
  const ans = read(tT);
  const c: Case = { id: `w${seed}`, at: iso(tc), horizon_minutes: (tT - tc) * 60, target: iso(tT), levels: [0.05, 0.25, 0.5, 0.75, 0.95], game: { release_date: '2026-11-19' }, trophies, history };
  return { c, answer: ans.rates };
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

export function pinball(q: number[], y: number): number {
  const L = [0.05, 0.25, 0.5, 0.75, 0.95];
  let s = 0;
  for (let j = 0; j < 5; j++) { const d = logitPct(y) - logitPct(q[j]); s += 2 * Math.max(L[j] * d, (L[j] - 1) * d); }
  return s / 5;
}

const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const inv = (l: number): number => 100 / (1 + Math.exp(-l));
export function persistence(c: Case): number[][] {
  const last = c.history[c.history.length - 1];
  const sp = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
  return c.trophies.map((_, i) => Z.map((z) => inv(logitPct(last.rates[i] ?? 0) + z * sp)));
}
export function curve(c: Case): number[][] {
  const tc = Date.parse(c.at);
  const xs = (iso: string): number => Math.log(1 + (Date.parse(iso) - T0) / 3_600_000);
  const recent = c.history.filter((h) => Date.parse(h.at) >= tc - 24 * 3_600_000);
  const xT = xs(c.target);
  const sp = 0.5 * Math.sqrt(c.horizon_minutes / 1440);
  return c.trophies.map((_, i) => {
    const pts = recent.filter((h) => h.rates[i] !== null).map((h) => [xs(h.at), logitPct(h.rates[i] as number)]);
    const n = pts.length;
    const mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
    let sxx = 0, sxy = 0; for (const [x, y] of pts) { sxx += (x - mx) ** 2; sxy += (x - mx) * (y - my); }
    const b = sxx > 0 ? sxy / sxx : 0; const a = my - b * mx;
    let rss = 0; for (const [x, y] of pts) rss += (y - a - b * x) ** 2;
    const s = Math.sqrt(sp * sp + rss / Math.max(1, n - 2));
    return Z.map((z) => inv(a + b * xT + z * s));
  });
}

function main(): void {
  const worlds = Number(process.argv[2] ?? 30);
  const seed0 = Number(process.argv[3] ?? 1);
  if (process.argv[4]) Object.assign(P, JSON.parse(process.argv[4]));
  const tot = { mine: 0, pers: 0, curve: 0 }; let cov = 0, n = 0;
  const perWorld: string[] = [];
  for (let w = 0; w < worlds; w++) {
    const { c, answer } = makeWorld(seed0 + w);
    const mine = forecast(c); const pe = persistence(c); const cu = curve(c);
    let m = 0, p = 0, u = 0;
    c.trophies.forEach((t, i) => {
      const y = answer[i]; const q = mine.find((r) => r.id === t.id)!.q;
      m += pinball(q, y); p += pinball(pe[i], y); u += pinball(cu[i], y);
      if (q[0] <= y && y <= q[4]) cov++; n++;
    });
    const K = c.trophies.length;
    tot.mine += m / K; tot.pers += p / K; tot.curve += u / K;
    perWorld.push(`${c.id} mine ${(m / K).toFixed(3)} pers ${(p / K).toFixed(3)} curve ${(u / K).toFixed(3)} N ${c.history[c.history.length - 1].players_tracked}`);
  }
  if (process.env.VERBOSE) console.log(perWorld.join('\n'));
  console.log(`worlds ${worlds}: mine ${(tot.mine / worlds).toFixed(4)}  persistence ${(tot.pers / worlds).toFixed(4)}  curve ${(tot.curve / worlds).toFixed(4)}  cov90 ${(cov / n).toFixed(3)}`);
}

if ((process.argv[1] ?? '').split('\\').join('/').endsWith('lab/sim.ts')) main();
