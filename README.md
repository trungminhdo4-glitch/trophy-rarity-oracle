# trophy-rarity-oracle

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/trungminhdo4-glitch/trophy-rarity-oracle/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/trungminhdo4-glitch/trophy-rarity-oracle/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PS5 trophy's Exophase earn rate one week after launch, as five quantiles on the logit scale.

How it works: a rate is earners / players tracked. The history shows when each batch of players
appeared on Exophase, so `main.ts` models every trophy as "share p of players earn it after a lognormal
wait (median e^mu hours, spread s) since appearing", fits a Bayesian grid over (mu, s, p) to the
trophy's observed history, extrapolates future player arrivals from the recent growth of the tracked
count (three decay scenarios), and reads the quantiles of the posterior predictive mixture, blurred
by a model-error kernel. Deterministic, standard library only, about a second per case.

`lab/sim.ts` is an agent-based launch simulator (diurnal play, quitting, irregular Exophase refreshes)
used to calibrate the knobs against the two house baselines.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
node lab/sim.ts 30 1   # calibration on 30 simulated launches
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
