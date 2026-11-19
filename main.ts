// Trophy Rarity Oracle: a structural forecaster for GTA VI's Exophase trophy earn rates.
//
// Idea: the page's rate for trophy k is earners_k / players_tracked. Players_tracked is observed at every
// Exophase refresh, so the history tells us when each batch of players appeared. A player who appeared at
// time a earns trophy k by time t with probability p_k * G((t - a) ; mu_k, s_k), where G is a lognormal CDF
// of wall-clock hours since appearing (p_k is the share of players that ever get it). So
//
//   earners_k(t) = c0_k + p_k * sum_j dN_j * G(t - a_j),   rate_k(t) = earners_k(t) / N(t)
//
// with c0_k the launch-night cohort's earners (held fixed). For every trophy we put a grid over
// (mu, s, p), weigh each point by how well it reproduces the trophy's observed history (tempered
// Gaussian on the count scale, with the page's 2-decimal rounding), extrapolate future arrivals of
// players from the recent growth of players_tracked (three decay scenarios), and read the posterior
// predictive mixture of the target's logit rate, blurred by a model-error kernel, at the five levels.
//
// Node 22, standard library only, deterministic, no network.
//   mkdir -p out && MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=out node main.ts

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

export const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];

export interface Trophy { id: number; name: string; description: string; tier: string; points: number; secret: boolean }
export interface Read { slot: string; at: string; players_tracked: number; trophies_earned?: number; rates: Array<number | null> }
export interface Case { id: string; at: string; horizon_minutes: number; target: string; levels: number[]; game?: { release_date?: string }; trophies: Trophy[]; history: Read[] }
export interface Row { id: number; q: number[] }

/** Tunable knobs (the lab's simulator calibrates them). */
export const P = {
  lagHours: 1.0,        // hours a fresh player has already played when Exophase first counts them
  temper: 3,            // effective number of independent observations per trophy (sim sweep: 2..4 best)
  relErr: 0.04,         // relative model error on the count scale
  overdisp: 4.0,        // count noise variance per expected earner
  kernel0: 0.12,        // model-error kernel on the logit scale, at zero predicted move
  kernelMove: 0.30,     // ... plus this share of the predicted logit move
  kernelHorizon: 0.10,  // ... plus this per log-ratio of target to checkpoint time
  decayWeights: [0.25, 0.5, 0.25],
  decayMult: [0.65, 1.0, 1.35],
};

const MU = range(Math.log(0.4), Math.log(4000), 0.12);
const S = [0.45, 0.75, 1.05, 1.4, 1.9];
const PL = range(-7, 5, 0.25); // logit of p

function range(a: number, b: number, step: number): number[] {
  const out: number[] = [];
  for (let x = a; x <= b + 1e-9; x += step) out.push(x);
  return out;
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
export const logitPct = (pct: number): number => { const c = clampPct(pct); return Math.log(c / (100 - c)); };
const invLogitPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round2 = (x: number): number => Math.round(x * 100) / 100;
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

/** Standard normal CDF (Abramowitz-Stegun 7.1.26, |err| < 1.5e-7). */
function phi(x: number): number {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
}

interface Snap { t: number; n: number; rates: Array<number | null> }

/** Distinct Exophase refreshes: a read that repeats the previous page adds nothing. */
function snapshots(c: Case, t0: number): Snap[] {
  const out: Snap[] = [];
  let prevT = -Infinity;
  for (const h of c.history) {
    const t = (Date.parse(h.at) - t0) / 3_600_000;
    const last = out[out.length - 1];
    const same = last !== undefined && last.n === h.players_tracked && h.rates.every((r, i) => r === last.rates[i]);
    if (!same) {
      // The refresh happened somewhere since the previous read: put it a little before this one.
      const when = Number.isFinite(prevT) ? Math.max(prevT, t - Math.min(0.25, (t - prevT) / 2)) : t;
      out.push({ t: when, n: h.players_tracked, rates: h.rates });
    }
    prevT = t;
  }
  return out;
}

/** Future arrivals as (time, count) bins from the checkpoint to the target, per decay scenario. */
function futureArrivals(snaps: Snap[], tc: number, tT: number): Array<Array<[number, number]>> {
  const nAt = (t: number): number => {
    let n = snaps[0].n;
    for (const s of snaps) if (s.t <= t) n = s.n;
    return n;
  };
  const tl = snaps[snaps.length - 1].t;
  const w = 24;
  const a1 = Math.max(0, nAt(tl) - nAt(tl - w)) / w;
  const a2 = Math.max(0, nAt(tl - w) - nAt(tl - 2 * w)) / w;
  let ratio = a2 > 0 && snaps[0].t < tl - 2 * w + 6 ? a1 / a2 : 0.6;
  if (!Number.isFinite(ratio)) ratio = 0.6;
  ratio = Math.min(1.0, Math.max(0.2, ratio));
  return P.decayMult.map((m) => {
    const r = Math.min(1.1, Math.max(0.1, ratio * m));
    const bins: Array<[number, number]> = [];
    const step = 2;
    for (let t = tl; t < tT; t += step) {
      const mid = Math.min(t + step / 2, (t + tT) / 2);
      const len = Math.min(step, tT - t);
      bins.push([mid, a1 * Math.pow(r, (mid - (tl - w / 2)) / 24) * len]);
    }
    return bins;
  });
}

export function forecast(c: Case): Row[] {
  const t0 = Date.parse(`${c.game?.release_date ?? '2026-11-19'}T00:00:00Z`);
  const tc = (Date.parse(c.at) - t0) / 3_600_000;
  const tT = (Date.parse(c.target) - t0) / 3_600_000;
  const snaps = snapshots(c, t0);
  const K = c.trophies.length;
  const base = snaps[0];
  // Arrival batches after the first snapshot.
  const arr: Array<[number, number]> = [];
  for (let j = 1; j < snaps.length; j++) arr.push([snaps[j].t, Math.max(0, snaps[j].n - snaps[j - 1].n)]);
  // Observations used in the fit: thin to at most 60, always keeping the latest ones.
  let obs = snaps.slice(1);
  if (obs.length > 60) {
    const keep = new Set<number>();
    for (let i = obs.length - 20; i < obs.length; i++) keep.add(i);
    for (let k = 0; k < 40; k++) keep.add(Math.floor((k * (obs.length - 20)) / 40));
    obs = obs.filter((_, i) => keep.has(i));
  }
  const fut = futureArrivals(snaps, tc, tT);
  const nLast = snaps[snaps.length - 1].n;
  const nT = fut.map((bins) => nLast + bins.reduce((s, b) => s + b[1], 0));

  // Per (mu, s): the reach S(t) = sum dN_j G(t - a_j) at every observation and, per scenario, at the target.
  const G = (dt: number, mu: number, s: number): number => (dt <= -P.lagHours ? 0 : phi((Math.log(dt + P.lagHours) - mu) / s));
  const grid: Array<{ mu: number; s: number; reach: number[]; reachT: number[] }> = [];
  for (const s of S) for (const mu of MU) {
    const reach = obs.map((o) => { let sum = 0; for (const [a, d] of arr) if (a <= o.t) sum += d * G(o.t - a, mu, s); return sum; });
    let past = 0;
    for (const [a, d] of arr) past += d * G(tT - a, mu, s);
    const reachT = fut.map((bins) => { let sum = past; for (const [a, d] of bins) sum += d * G(tT - a, mu, s); return sum; });
    grid.push({ mu, s, reach, reachT });
  }
  const temperFor = (m: number): number => Math.min(1, P.temper / Math.max(1, m));
  const kernelBase = P.kernel0 + P.kernelHorizon * Math.log(Math.max(1, (tT + P.lagHours) / Math.max(1, tc + P.lagHours)));

  const rows: Row[] = [];
  for (let k = 0; k < K; k++) {
    const t = c.trophies[k];
    const lastRate = lastKnown(snaps, k);
    const c0 = ((base.rates[k] ?? 0) * base.n) / 100;
    const ks: number[] = [];
    for (let i = 0; i < obs.length; i++) if (obs[i].rates[k] !== null && obs[i].rates[k] !== undefined) ks.push(i);
    const tw = temperFor(ks.length);
    // Log weights of every grid point, and its predicted target logit per scenario.
    const comps: Array<{ lw: number; l: number }> = [];
    for (const g of grid) {
      for (const pl of PL) {
        const p = 1 / (1 + Math.exp(-pl));
        let ll = 0;
        for (const i of ks) {
          const o = obs[i];
          const m = c0 + p * g.reach[i];
          const y = ((o.rates[k] as number) * o.n) / 100;
          const q = (0.01 * o.n) / 100; // the page's rounding step in earners
          const v = P.overdisp * Math.max(m, 1) + (q * q) / 12 + (P.relErr * m) ** 2 + 0.25;
          ll += -0.5 * ((y - m) ** 2 / v) - 0.5 * Math.log(v);
        }
        const prior = -0.5 * ((pl - 0.5) / 3) ** 2;
        const lw = tw * ll + prior;
        for (let sc = 0; sc < fut.length; sc++) {
          const rate = (100 * (c0 + p * g.reachT[sc])) / nT[sc];
          comps.push({ lw: lw + Math.log(P.decayWeights[sc]), l: logitPct(round2(Math.min(100, rate))) });
        }
      }
    }
    rows.push({ id: t.id, q: mixtureQuantiles(comps, logitPct(lastRate), kernelBase) });
  }
  return rows;
}

function lastKnown(snaps: Snap[], k: number): number {
  for (let j = snaps.length - 1; j >= 0; j--) { const r = snaps[j].rates[k]; if (r !== null && r !== undefined) return r; }
  return 0.01;
}

/** Quantiles (percent) of a weighted mixture of logits, each blurred by a Gaussian kernel. */
function mixtureQuantiles(comps: Array<{ lw: number; l: number }>, lastL: number, kernelBase: number): number[] {
  let mx = -Infinity;
  for (const c of comps) if (c.lw > mx) mx = c.lw;
  // Collapse onto a fine logit lattice to keep the CDF cheap.
  const cells = new Map<number, number>();
  let tot = 0;
  for (const c of comps) {
    const w = Math.exp(c.lw - mx);
    if (w < 1e-12) continue;
    const key = Math.round(c.l * 50);
    cells.set(key, (cells.get(key) ?? 0) + w);
    tot += w;
  }
  const pts = [...cells.entries()].map(([key, w]) => ({ l: key / 50, w: w / tot })).sort((a, b) => a.l - b.l);
  let mean = 0;
  for (const p of pts) mean += p.w * p.l;
  const sd = Math.max(kernelBase, P.kernel0 + P.kernelMove * Math.abs(mean - lastL));
  const cdf = (x: number): number => { let s = 0; for (const p of pts) s += p.w * phi((x - p.l) / sd); return s; };
  const lo = pts[0].l - 6 * sd - 1;
  const hi = pts[pts.length - 1].l + 6 * sd + 1;
  const out: number[] = [];
  for (const tau of LEVELS) {
    let a = lo, b = hi;
    for (let it = 0; it < 50; it++) { const m = (a + b) / 2; if (cdf(m) < tau) a = m; else b = m; }
    out.push(round4(invLogitPct((a + b) / 2)));
  }
  for (let j = 1; j < out.length; j++) if (out[j] < out[j - 1]) out[j] = out[j - 1];
  return out.map((x) => Math.min(100, Math.max(0, x)));
}

function main(): void {
  const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  mkdirSync(OUTPUT_DIR, { recursive: true });
  for (const file of readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort()) {
    const id = file.slice(0, -'.in'.length);
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    let rows: Row[];
    try {
      rows = forecast(c);
    } catch {
      rows = fallback(c);
    }
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies: rows })}\n`);
  }
}

/** A safe forecast if the model ever throws: last rate with a random-walk spread. */
export function fallback(c: Case): Row[] {
  const last = c.history[c.history.length - 1];
  const spread = 0.8 * Math.sqrt(c.horizon_minutes / 1440) + 0.1;
  const Z = [-1.645, -0.674, 0, 0.674, 1.645];
  return c.trophies.map((t, i) => {
    const l = logitPct(last?.rates[i] ?? 0.01);
    return { id: t.id, q: Z.map((z) => round4(invLogitPct(l + z * spread))) };
  });
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) main();
