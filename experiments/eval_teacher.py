"""Fly the teacher on N seeds in several worlds and print landed rate, outcomes, score."""
import sys, collections
from multiprocessing import Pool
import numpy as np
from lander.env import World, run_episode, make_env
from lander.teacher import Teacher, load_params

WORLDS = {'calm': World(wind=0, turbulence=0), 'wind15+turb': World(), 'goal(boulder)': World(boulder=True),
          'calm+boulder': World(wind=0, turbulence=0, boulder=True)}

def fly(args):
    name, seed, p = args
    return name, run_episode(Teacher(p or load_params()), seed, WORLDS[name])

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    start = 3_000_000_000
    with Pool(6) as pool:
        res = pool.map(fly, [(w, start + i, None) for w in WORLDS for i in range(n)], chunksize=10)
    for w in WORLDS:
        rs = [r for k, r in res if k == w]
        c = collections.Counter(r['outcome'] for r in rs)
        print(f"{w:15s} landed {np.mean([r['landed'] for r in rs]):6.1%}  score {np.mean([r['reward'] for r in rs]):6.1f}  {dict(c)}  offpad {sum(r['rested'] and not r['landed'] for r in rs)}")
