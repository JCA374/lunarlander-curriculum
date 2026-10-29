"""Paired comparison on the same analysis seeds: champion(s) vs the teacher pilot, in a named world.
    python -m experiments.paired WORLD N champ1.pkl [champ2.pkl ...]     WORLD: steady15 | wind | goal | gusts"""
import sys, pickle
from multiprocessing import Pool
import neat, numpy as np
from lander.env import World, run_episode, make_env
from lander.neatnet import NeatPolicy, SeededGenome
from lander.teacher import Teacher, load_params
from train import ANALYSIS
WORLDS = {'steady15': World(wind=15, steady=1.0), 'gusts': World(turbulence=0), 'wind': World(), 'goal': World(boulder=True)}
_W = {}
def init():
    _W['cfg'] = neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-config.ini')
    _W['env'] = make_env()
def fly(a):
    who, w, s = a
    pol = Teacher(load_params()) if who == 'teacher' else NeatPolicy(pickle.load(open(who, 'rb')), _W['cfg'])
    return run_episode(pol, s, WORLDS[w], env=_W['env'])['landed']
if __name__ == '__main__':
    w, n, champs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    who = ['teacher'] + champs
    seeds = [ANALYSIS + 100_000 + i for i in range(n)]
    with Pool(6, initializer=init) as p:
        res = {x: np.array(p.map(fly, [(x, w, s) for s in seeds], chunksize=10)) for x in who}
    rng = np.random.default_rng(0)
    for x in champs:
        d = res[x].astype(float) - res['teacher']
        boots = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
        print(f"{w}: {x.split('/')[-1]} {res[x].mean():.1%} vs teacher {res['teacher'].mean():.1%}; "
              f"difference {d.mean():+.1%} [95% CI {np.percentile(boots, 2.5):+.1%}, {np.percentile(boots, 97.5):+.1%}] on {n} paired flights")
