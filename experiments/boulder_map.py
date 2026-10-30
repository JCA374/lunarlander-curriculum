"""Where do flights fail? Landing rate by the boulder's height (bottom above the pad) and sideways offset.
    python -m experiments.boulder_map N [champ.pkl]      (pilot = the current tuned gains)"""
import sys, pickle
from multiprocessing import Pool
import neat, numpy as np
from lander.env import GOAL_WORLD, run_episode, make_env, boulder_spec, W, PAD_Y
from lander.neatnet import NeatPolicy, SeededGenome
from lander.teacher import Teacher, load_params
from train import ANALYSIS
_W = {}
def init(path):
    _W['env'] = make_env()
    if path:
        cfg = neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'neat-config.ini')
        _W['pol'] = NeatPolicy(pickle.load(open(path, 'rb')), cfg)
    else:
        _W['pol'] = Teacher(load_params())
def fly(s):
    r = run_episode(_W['pol'], s, GOAL_WORLD, env=_W['env'])
    x, y, rad = boulder_spec(s)
    return y - rad - PAD_Y, abs(x - W / 2), rad, r['landed'], r['outcome']
if __name__ == '__main__':
    n = int(sys.argv[1]); path = sys.argv[2] if len(sys.argv) > 2 else None
    with Pool(8, initializer=init, initargs=(path,)) as p:
        rs = p.map(fly, [ANALYSIS + 400_000 + i for i in range(n)], chunksize=10)
    gap = np.array([r[0] for r in rs]); off = np.array([r[1] for r in rs]); ok = np.array([r[3] for r in rs])
    print(f"{path or 'pilot'}: overall {ok.mean():.1%} on {n}")
    print("rows: gap between boulder bottom and pad (m); cols: boulder centre offset from the pad centre (m)")
    gb, ob = [1.5, 2.5, 3.5, 4.5, 6.5], [0, 0.75, 1.5, 2.5]
    print('gap \\ offset ' + ''.join(f'{f"{a}-{b}":>14s}' for a, b in zip(ob, ob[1:])))
    for a, b in zip(gb, gb[1:]):
        cells = []
        for c, d in zip(ob, ob[1:]):
            m = (gap >= a) & (gap < b) & (off >= c) & (off < d)
            cells.append(f"{ok[m].mean():6.0%} (n={m.sum():3d})" if m.sum() else ' ' * 14)
        print(f'{a}-{b} m'.ljust(13) + ''.join(f'{x:>14s}' for x in cells))
