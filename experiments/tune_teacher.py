"""Tune the teacher's gains with CMA-ES on training seeds.
    python -m experiments.tune_teacher GENS [goal] [OUT]
default: wind+turbulence, half the flights with the boulder (the first tuning). 'goal': every flight is the final task
(wind 15, turbulence 1.5, boulder), starting from the current tuned gains. OUT: where to write the gains."""
import json, sys
from multiprocessing import Pool
import cma, numpy as np
from lander.env import World, run_episode
from lander.teacher import Teacher, P, load_params

GOAL = len(sys.argv) > 2 and sys.argv[2] == 'goal'
OUT = sys.argv[3] if len(sys.argv) > 3 else 'lander/teacher_params.json'
START = load_params() if GOAL else P
KEYS = list(P)
X0 = np.array([START[k] for k in KEYS])
WORLD_W, WORLD_B = World(), World(boulder=True)

def fly(args):
    p, seed, b = args
    r = run_episode(Teacher(p), seed, WORLD_B if (b or GOAL) else WORLD_W)
    return r['landed'] + 0.002 * r['reward']

def score(pool, x, seeds):
    p = dict(zip(KEYS, np.abs(x) * np.maximum(np.abs(X0), 0.05)))
    return np.mean(pool.map(fly, [(p, s, i % 2) for i, s in enumerate(seeds)], chunksize=8))

if __name__ == '__main__':
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    es = cma.CMAEvolutionStrategy(np.ones(len(KEYS)), 0.2 if GOAL else 0.3, {'popsize': 16, 'seed': 1, 'verbose': -9})
    best = None
    with Pool(6) as pool:
        for g in range(gens):
            seeds = list(range(g * 1000 + (600_000_000 if GOAL else 0), g * 1000 + (600_000_000 if GOAL else 0) + 160))
            xs = es.ask()
            fs = [score(pool, x, seeds) for x in xs]
            es.tell(xs, [-f for f in fs])
            m = es.mean
            print(g, f"best {max(fs):.3f} mean-of-pop {np.mean(fs):.3f}", flush=True)
    p = dict(zip(KEYS, (np.abs(es.mean) * np.maximum(np.abs(X0), 0.05)).round(4).tolist()))
    json.dump(p, open(OUT, 'w'), indent=1)
    print(p)
