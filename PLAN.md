# Lunar Lander in wind, turbulence and one more obstacle: a curriculum (plan)

**Status:** planned 2026-10-29 at the end of the Bipedal CMA-ES project. No code yet: start here.
Repo (to create): https://github.com/JCA374/lunarlander-curriculum. Site (GitHub Pages from `main:/docs`):
https://jca374.github.io/lunarlander-curriculum/

How to work: follow [METHOD.md](METHOD.md) (2-hour check-ins, one change at a time with a written reason, verify on
fresh flights, a log page, a podcast for every major event). The podcast is the user-level skill
`project-podcast` (`podcast.conf` is already in this folder).

## 1. Goal

Gymnasium's LunarLander, "advanced" settings, continuous or discrete engines (decide at kickoff; discrete
matches the earlier projects):

- **Wind** `wind_power = 15` (gusts sideways) and **turbulence** `turbulence_power = 1.5` (a twisting torque).
- **One more obstacle** (to confirm with the user at kickoff, see §3, stage 5). Recommended: **a boulder on the
  approach**, a static Box2D body between the start and the pad that the lander must fly around or over. The built-in
  alternative is **variable gravity** (`gravity` from −12 to −8 per flight).

**Target:** on **1,000 fresh flights** with all of it turned on: **≥ 95% landed between the flags, mean score ≥ 200**,
0% timeouts. Report the landed rate with a confidence interval, and compare paired with the best earlier lander on the same seeds.

Baseline to beat (`../lunarlander-neat-wind`, wind 15 + turbulence 1.5, no obstacle): NEAT run J
**91% landed, score 218** [213, 223] (1,000 flights); run D 85%, 191. A hand-written controller with a drift
estimate reached 94% (216); without the drift estimate, 72%.

## 2. Model and search

- **CMA-ES** (pycma, ask/tell) over an MLP, at least one hidden layer: 8 observations + extra inputs → 32 tanh → 4 (discrete:
  argmax) or 2 (continuous). Start from the Bipedal setup: λ 48, K flights per candidate with common random numbers, σ0 0.5,
  IPOP restarts, the plateau rule, sep-CMA if the network gets large.
- **Extra inputs that proved necessary:** a **drift sensor** (a smoothed sideways acceleration estimate from the
  observations, `DriftSensor` in `../lunarlander-neat-wind/neat_lander/env.py`) and **|x|**. For the boulder stage, add
  the boulder's relative position (dx, dy), or a few lidar-like range rays if the boulder moves around.
- Option if it plateaus: a small recurrent state (memory of the gusts), then a wider network (Net2Net widen worked in Bipedal; deepening did not).

## 3. The curriculum

Each stage starts from the previous stage's champion and moves on when its check passes (on 100 fresh flights) or its
generation cap runs out. Each stage's reward and expected outcome go on the site **before** it starts.

| # | Stage | World | Training reward (on top of the env score) | Pass check (100 fresh flights) |
|---|---|---|---|---|
| 1 | **Land calm** | no wind | + centre bonus (up to 50, falling to 0 at the flags) + landing bonus 100 | ≥ 97% landed, score ≥ 230 |
| 2 | **Steady push** | a constant sideways force, 30-100% of 15, either way (`steady_push` from the wind project) | same | ≥ 95% landed |
| 3 | **Gusts** | the real wind, `wind_power` ramped 5 → 10 → 15 inside the stage | same | ≥ 93% landed at 15 |
| 4 | **Gusts + turbulence** | wind 15, `turbulence_power` ramped 0.5 → 1.5 | same | ≥ 93% landed, score ≥ 200 |
| 5 | **The obstacle** | wind + turbulence + the boulder (random size and place on the approach) *or* gravity −12 … −8 | + touching the boulder costs a crash (−100); a small bonus for clearance | ≥ 90% landed, 0 boulder hits |
| 6 | **Polish** | everything on, the real env score + landing bonus only | — | the goal (§1), on 1,000 flights |

**Training wheels** (what worked best in Bipedal): in stages 2-4, add a stabilising torque on the lander's angle,
`support × (k1·angle + k2·angular velocity)`, that fades from 1.0 to 0 across the stage. The pass check is always flown
**without** it.

**Traps to watch for** (seen in the earlier projects): hovering until the timeout to avoid crashing; landing off the
pad on purpose because it's safe; touching down fast for the leg contact points; a lander that gets reckless
because the landing bonus outweighs the crash cost. Check the timeout rate, the off-pad rate and the touchdown speed at every check-in.

## 4. Lessons carried over

From **lunarlander-neat-wind**:

- **Seed the wind.** Gymnasium draws the gust phase from the global numpy RNG, not the reset seed. Set
  `env.unwrapped.wind_idx` and `torque_idx` from the episode seed (`seed_wind`) so one seed always means the same wind. Without that, common random numbers and resume don't work.
- **Fly pad-first:** reach the flat pad horizontally, then descend. The ground outside the flags is hilly, so off-pad touchdowns slide or tip over.
- **Sensing the drift matters more than network size.** NEAT that only saw velocity stalled near 85%.
- **A steady-push curriculum without a wind sensor was superseded** (runs G and H): give the sensor first, then the curriculum.
- **Few flights make selection lucky.** 5 flights per genome couldn't tell an 85% lander from a 95% one. Use 8 or more,
  plus re-flying the top candidates on 60 more.
- **Validation seed ranges** were 1e9 (validation), 2e9 (test) and 3e9 (analysis). Keep training, selection, check, test and
  comparison seeds separate.

From **bipedal-CMA-ES** (see its METHOD.md and TRAINING_LOG.md):

- **The winner's curse:** champions chosen on 20 tests looked about 15 points better than they were, three times
  over. Choose champions from the CMA mean, check a goal on ≥ 50 fresh cases and claim "better" only from a paired 100-case comparison.
- **Archive every champion.** Don't let a later snapshot overwrite a verified best.
- **Determinism first:** "same seed twice gives the same reward" is the first test (Box2D world reuse broke it there).
- **Shaping terms make traps.** Prefer training wheels that fade out, and keep the shaping small and few.
- **SIGTERM saves a checkpoint. Use `setsid nohup`, and never `pkill -f` a pattern that's also in your own command.**

## 5. First steps (day 1)

0. If the GitHub repo doesn't exist yet: `gh repo create JCA374/lunarlander-curriculum --public --source . --push`, then
   enable Pages from `main:/docs` (`gh api -X POST repos/JCA374/lunarlander-curriculum/pages -f "source[branch]=main" -f "source[path]=/docs"`).
1. `.venv`: `numpy<2`, `gymnasium[box2d]==0.29.1`, `cma`, `pytest`, `Pillow`, `opencv-python` (as in Bipedal).
2. Tests first: same seed twice gives the same reward (with wind and turbulence seeded), flatten round-trip, ≥ 1 hidden layer,
   no seed-range overlap, smoke train → resume continues the generation and the CMA state.
3. Port from `../../bi-pedal/bipedal-CMA-ES`: `train.py` (CMA loop, HOF on the mean, champion archive, overrides, SIGTERM),
   `curriculum.sh` style drivers, `plateau_watch.sh`, `curriculum_publish.sh`, `make_curriculum_log.py` (the newest-first log
   with a replay per event), `dashboard.py`, `experiments/compare.py` / `goal_check.py`. Port the env wrapper, `seed_wind`, `steady_push` and `DriftSensor`
   from `../lunarlander-neat-wind/neat_lander/env.py`.
4. Build the boulder as a wrapper that adds a static body to `env.unwrapped.world` after reset (and draws it). Test that a
   hit ends the flight as a crash.
5. Site: the home page, lessons (the task, wind and turbulence, sensing drift, CMA-ES short recap, the curriculum), a Curriculum log tab,
   a Podcast tab. Episode 1: the plan.
6. Start stage 1, set up the 2-hour check-in cron, and agree on a deadline with the user.

## 6. Hardware

i5-8250U, 4 cores / 8 threads, 7.6 GB RAM, no GPU. Training gets 6 workers; narration and video run with `nice`.
