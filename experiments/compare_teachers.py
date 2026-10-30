"""Paired comparison of two pilot gain files on the same analysis seeds.
    python -m experiments.compare_teachers A.json B.json [N]"""
import sys, json, collections
from multiprocessing import Pool
import numpy as np
from lander.env import GOAL_WORLD, World, run_episode
from lander.teacher import Teacher, P
from train import ANALYSIS
WORLDS = {'goal': GOAL_WORLD, 'wind': World()}
def fly(a):
    path, w, s = a
    r = run_episode(Teacher(dict(P, **json.load(open(path)))), s, WORLDS[w])
    return 'landed' if r['landed'] else ('off-pad' if r['rested'] else r['outcome'])
if __name__ == '__main__':
    A, B = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    rng = np.random.default_rng(0)
    with Pool(8) as p:
        for w in WORLDS:
            seeds = [ANALYSIS + 300_000 + i for i in range(n)]
            ra = p.map(fly, [(A, w, s) for s in seeds], chunksize=10)
            rb = p.map(fly, [(B, w, s) for s in seeds], chunksize=10)
            a = np.array([r == 'landed' for r in ra], float); b = np.array([r == 'landed' for r in rb], float)
            d = b - a
            boots = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
            print(f"{w}: {A.split('/')[-1]} {a.mean():.1%} {dict(collections.Counter(ra))}")
            print(f"{w}: {B.split('/')[-1]} {b.mean():.1%} {dict(collections.Counter(rb))}")
            print(f"{w}: difference {d.mean():+.1%} [95% CI {np.percentile(boots, 2.5):+.1%}, {np.percentile(boots, 97.5):+.1%}] on {n} paired flights")
