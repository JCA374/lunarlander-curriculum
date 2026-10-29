"""Tune the teacher's gains with CMA-ES on training seeds (wind+turbulence, half with the boulder)."""
import json, sys
from multiprocessing import Pool
import cma, numpy as np
from lander.env import World, run_episode
from lander.teacher import Teacher, P

KEYS = list(P)
X0 = np.array([P[k] for k in KEYS])
WORLD_W, WORLD_B = World(), World(boulder=True)

def fly(args):
    p, seed, b = args
    r = run_episode(Teacher(p), seed, WORLD_B if b else WORLD_W)
    return r['landed'] + 0.002 * r['reward']

def score(pool, x, seeds):
    p = dict(zip(KEYS, np.abs(x) * np.maximum(np.abs(X0), 0.05)))
    return np.mean(pool.map(fly, [(p, s, i % 2) for i, s in enumerate(seeds)], chunksize=8))

if __name__ == '__main__':
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    es = cma.CMAEvolutionStrategy(np.ones(len(KEYS)), 0.3, {'popsize': 16, 'seed': 1, 'verbose': -9})
    best = None
    with Pool(6) as pool:
        for g in range(gens):
            seeds = list(range(g * 1000, g * 1000 + 160))     # fresh training seeds per generation
            xs = es.ask()
            fs = [score(pool, x, seeds) for x in xs]
            es.tell(xs, [-f for f in fs])
            m = es.mean
            print(g, f"best {max(fs):.3f} mean-of-pop {np.mean(fs):.3f}", flush=True)
    p = dict(zip(KEYS, (np.abs(es.mean) * np.maximum(np.abs(X0), 0.05)).round(4).tolist()))
    json.dump(p, open('lander/teacher_params.json', 'w'), indent=1)
    print(p)
