"""Why flights fail: outcome breakdown (boulder hit, crash, off-pad rest, timeout) for policies on the same analysis seeds.
    python -m experiments.failures N champ.pkl [...]          (always includes the teacher)"""
import sys, pickle, collections
from multiprocessing import Pool
import neat, numpy as np
from lander.env import GOAL_WORLD, run_episode, make_env, boulder_spec, W
from lander.neatnet import NeatPolicy, SeededGenome
from lander.teacher import Teacher, load_params
from train import ANALYSIS
_W = {}
def init():
    _W['cfg'] = neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-config.ini')
    _W['env'] = make_env()
def fly(a):
    who, s = a
    pol = Teacher(load_params()) if who == 'teacher' else NeatPolicy(pickle.load(open(who, 'rb')), _W['cfg'])
    r = run_episode(pol, s, GOAL_WORLD, env=_W['env'])
    kind = 'landed' if r['landed'] else ('off-pad rest' if r['rested'] else r['outcome'])
    bx, by, br = boulder_spec(s)
    return kind, abs(bx - W / 2), r['touch_speed']
if __name__ == '__main__':
    n, who = int(sys.argv[1]), ['teacher'] + sys.argv[2:]
    seeds = [ANALYSIS + 200_000 + i for i in range(n)]
    with Pool(6, initializer=init) as p:
        for w in who:
            rs = p.map(fly, [(w, s) for s in seeds], chunksize=10)
            c = collections.Counter(k for k, _, _ in rs)
            central = [k for k, off, _ in rs if off < 1.0]
            side = [k for k, off, _ in rs if off >= 1.0]
            print(f"{w.split('/')[-1]:24s} " + ', '.join(f"{k} {v / n:.1%}" for k, v in c.most_common()) +
                  f" | landed when boulder within 1 m of centre {np.mean([k == 'landed' for k in central]):.1%} (n={len(central)}), else {np.mean([k == 'landed' for k in side]):.1%}")
