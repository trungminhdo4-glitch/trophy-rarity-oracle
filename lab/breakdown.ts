// Lab: loss, median bias (truth minus median, logit) and 90% coverage per trophy family on simulated launches.
//   node lab/breakdown.ts [worlds] [seed] [knobs-json]
import { makeWorld, pinball } from './sim.ts';
import { forecast, logitPct, P } from '../main.ts';
const worlds = Number(process.argv[2] ?? 16), seed = Number(process.argv[3] ?? 401);
if (process.argv[4]) Object.assign(P, JSON.parse(process.argv[4]));
const agg: Record<string, { n: number; l: number; bias: number; cov: number }> = {};
let tot = 0, n = 0;
for (let w = 0; w < worlds; w++) {
  const { c, answer } = makeWorld(seed + w);
  const f = forecast(c);
  c.trophies.forEach((t, i) => {
    const cat = t.name.split(' ')[0]; const q = f[i].q; const L = pinball(q, answer[i]);
    const a = (agg[cat] ??= { n: 0, l: 0, bias: 0, cov: 0 });
    a.n++; a.l += L; a.bias += logitPct(answer[i]) - logitPct(q[2]); if (q[0] <= answer[i] && answer[i] <= q[4]) a.cov++;
    tot += L; n++;
  });
}
for (const [k, a] of Object.entries(agg)) console.log(k.padEnd(11), 'loss', (a.l / a.n).toFixed(3), 'bias', (a.bias / a.n).toFixed(3), 'cov', (a.cov / a.n).toFixed(2), 'n', a.n);
console.log('all', (tot / n).toFixed(4));
