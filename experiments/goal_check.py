"""The goal check: a champion on N unseen test-range flights (wind 15, turbulence 1.5, boulder), no help.

    python -m experiments.goal_check runs/A/champions/g01234_s5_passed.pkl --n 1000 [--note "..."]
Writes results/goal_<label>.json (shown in the site log) with a 95% Wilson interval on the landed rate.
"""
import argparse, collections, json, math, os, pickle, time
from multiprocessing import Pool
import neat, numpy as np
from lander.env import GOAL_WORLD, run_episode, make_env
from lander.neatnet import NeatPolicy, SeededGenome
from train import TEST

_W = {}

def _init(path):
    cfg = neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-config.ini')
    _W['pol'] = NeatPolicy(pickle.load(open(path, 'rb')), cfg)
    _W['env'] = make_env()

def fly(seed):
    r = run_episode(_W['pol'], seed, GOAL_WORLD, env=_W['env'])
    return {k: r[k] for k in ('seed', 'reward', 'landed', 'outcome', 'touch_speed', 'x_final')}

def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('champion'); ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--offset', type=int, default=0, help='use a different block of test seeds')
    ap.add_argument('--note', default=''); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    seeds = [TEST + a.offset + i for i in range(a.n)]
    with Pool(a.workers, initializer=_init, initargs=(a.champion,)) as p:
        rs = p.map(fly, seeds, chunksize=10)
    k = sum(r['landed'] for r in rs)
    out = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'champion': a.champion, 'n': a.n, 'landed': k / a.n,
           'ci': wilson(k, a.n), 'score': float(np.mean([r['reward'] for r in rs])),
           'outcomes': dict(collections.Counter(r['outcome'] for r in rs)),
           'offpad': sum(r['outcome'] == 'landed' and not r['landed'] for r in rs), 'note': a.note,
           'failed_seeds': [r['seed'] for r in rs if not r['landed']][:200]}
    os.makedirs('results', exist_ok=True)
    label = os.path.basename(a.champion).replace('.pkl', '') + (f'_o{a.offset}' if a.offset else '')
    json.dump(out, open(f'results/goal_{label}.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != 'failed_seeds'}, indent=1))
