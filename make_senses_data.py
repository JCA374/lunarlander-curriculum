"""Data for the "How the lander sees it" tab (docs/senses.html): docs/data/senses.json.

One recorded flight of the newest champion in the goal world, with, at every step: the 15 inputs the network
gets, and the truth it doesn't get (the wind force and turbulence torque actually applied, the exact position,
the terrain). The page draws both side by side.

    python make_senses_data.py [--run F]
"""

import argparse
import glob
import json
import math
import os
import pickle

from lander.env import GOAL_WORLD, INPUT_NAMES, Flight, make_env, W, H, PAD_Y
from lander.neatnet import NeatPolicy
from make_network_data import load_config

ROOT = os.path.dirname(os.path.abspath(__file__))


def true_forces(u):
    """The wind force and turbulence torque the next env.step applies (gymnasium's own formulas)."""
    if not u.enable_wind or u.legs[0].ground_contact or u.legs[1].ground_contact:
        return 0.0, 0.0
    k, j = u.wind_idx, u.torque_idx
    wind = math.tanh(math.sin(0.02 * k) + math.sin(math.pi * 0.01 * k)) * u.wind_power
    torque = math.tanh(math.sin(0.02 * j) + math.sin(math.pi * 0.01 * j)) * u.turbulence_power
    return wind, torque


def record(genome, config, seed):
    pol = NeatPolicy(genome, config)
    env = make_env()
    f = Flight(env, seed, GOAL_WORLD)
    u = env.unwrapped
    terrain = []
    for fx in u.moon.fixtures[1:]:                 # fixture 0 is the bottom edge of the world
        a, b = fx.shape.vertices
        terrain += [list(map(lambda t: round(t, 3), a)), list(map(lambda t: round(t, 3), b))]
    steps, done = [], False
    while not done:
        v = f.inputs()
        a = pol(v)
        wind, torque = true_forces(u)
        p = u.lander.position
        steps.append({'in': [round(float(t), 4) for t in v[:pol.n]], 'a': a,
                      'p': [round(p.x, 3), round(p.y, 3)], 'ang': round(float(u.lander.angle), 4),
                      'wind': round(wind, 3), 'torque': round(torque, 3)})
        done = f.step(a)
    p = u.lander.position
    steps.append({'in': [round(float(t), 4) for t in f.inputs()[:pol.n]], 'a': 0, 'p': [round(p.x, 3), round(p.y, 3)],
                  'ang': round(float(u.lander.angle), 4), 'wind': 0.0, 'torque': 0.0})
    r = f.result()
    env.close()
    return {'seed': seed, 'terrain': terrain, 'pad': [u.helipad_x1, u.helipad_x2, PAD_Y], 'W': W, 'H': H,
            'boulder': f.boulder, 'outcome': r['outcome'], 'landed': r['landed'], 'steps': steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None)
    a = ap.parse_args()
    runs = [d for d in sorted(glob.glob(os.path.join(ROOT, 'runs', '*')), key=os.path.getmtime)
            if glob.glob(os.path.join(d, 'champions', '*.pkl')) and not os.path.basename(d).startswith('test_')]
    run = a.run or os.path.basename(runs[-1])
    champs = sorted(glob.glob(os.path.join(ROOT, 'runs', run, 'champions', '*.pkl')), key=os.path.getmtime)
    config = load_config()
    genome = pickle.load(open(champs[-1], 'rb'))
    # the first of up to 40 analysis flights where the boulder sits near the straight line down and the lander lands:
    # the most telling flight to watch
    from lander.env import boulder_spec
    pick = None
    for i in range(40):
        seed = 3_000_777_001 + i
        if abs(boulder_spec(seed)[0] - W / 2) > 1.2:
            continue
        tr = record(genome, config, seed)
        pick = pick or tr
        if tr['landed']:
            pick = tr
            break
    pick.update(run=run, champion=os.path.basename(champs[-1]), input_names=INPUT_NAMES)
    json.dump(pick, open(os.path.join(ROOT, 'docs', 'data', 'senses.json'), 'w'), separators=(',', ':'))
    print(f"senses.json: run {run}, {pick['champion']}, seed {pick['seed']}, {pick['outcome']}, {len(pick['steps'])} steps")


if __name__ == '__main__':
    main()
