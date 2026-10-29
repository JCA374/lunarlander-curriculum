"""Data for the Network tab (docs/network.html): docs/data/network.json.

- seed: the cloned start network as a NEAT genome
- champions: every archived champion of the main run (genome, generation, stage, fresh-check result, change vs seed)
- trace: one recorded flight of the newest champion in the goal world: inputs, every node's value, action, position
- presets: interesting moments from that flight (for the "try it" sliders)
- config: the NEAT settings that shape evolution

    python make_network_data.py [--run D]
"""

import argparse
import glob
import json
import os
import pickle
import re

import neat
import numpy as np
from neat.graphs import feed_forward_layers

from lander.env import GOAL_WORLD, INPUT_NAMES, Flight, make_env, lander_xy
from lander.neatnet import SeededGenome, NeatPolicy, mlp_to_genome

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_NAMES = ['nothing', 'left engine', 'main engine', 'right engine']


def load_config():
    return neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       os.path.join(ROOT, 'neat-config.ini'))


def export(genome, config):
    gc = config.genome_config
    conns = [(k, c) for k, c in genome.connections.items() if c.enabled]
    layers = feed_forward_layers(gc.input_keys, gc.output_keys, [k for k, _ in conns])
    depth = {k: 0 for k in gc.input_keys}
    for i, layer in enumerate(layers):
        for n in layer:
            depth[n] = i + 1
    return {
        'nodes': [{'k': k, 'b': round(n.bias, 6), 'a': n.activation, 'r': n.response,
                   'd': depth.get(k, 1)} for k, n in sorted(genome.nodes.items())],
        'conns': [{'i': a, 'o': b, 'w': round(c.weight, 6)} for (a, b), c in conns],
        'n_disabled': sum(not c.enabled for c in genome.connections.values()),
    }


def diff(genome, seed):
    sw = {k: c.weight for k, c in seed.connections.items()}
    shared = [abs(c.weight - sw[k]) for k, c in genome.connections.items() if k in sw and c.enabled]
    new_conns = [k for k, c in genome.connections.items() if k not in sw and c.enabled]
    new_nodes = [k for k in genome.nodes if k not in seed.nodes]
    lost = [k for k in sw if k not in genome.connections or not genome.connections[k].enabled]
    return {'mean_dw': float(np.mean(shared)) if shared else 0.0, 'max_dw': float(np.max(shared)) if shared else 0.0,
            'new_conns': len(new_conns), 'new_nodes': len(new_nodes), 'lost_conns': len(lost),
            'new_node_keys': new_nodes}


def trace(genome, config, seed):
    pol = NeatPolicy(genome, config)
    env = make_env()
    f = Flight(env, seed, GOAL_WORLD)
    steps = []
    done = False
    while not done:
        v = f.inputs()
        a = pol(v)
        vals = pol.net.values
        x, y = lander_xy(env)
        steps.append({'in': [round(float(t), 5) for t in v[:pol.n]],
                      'h': {str(k): round(float(vals[k]), 3) for k in genome.nodes},
                      'a': a, 'p': [round(x, 2), round(y, 2)], 'ang': round(float(env.unwrapped.lander.angle), 3)})
        done = f.step(a)
    env.close()
    return {'seed': seed, 'boulder': f.boulder, 'outcome': f.result()['outcome'],
            'landed': f.result()['landed'], 'steps': steps}


def presets(tr):
    """A handful of telling moments from the flight, labelled by what is going on."""
    s = tr['steps']
    out = []

    def add(label, i, why):
        if 0 <= i < len(s):
            out.append({'label': label, 'step': i, 'why': why, 'in': s[i]['in']})
    add('The start', 2, 'High above the pad, just after the random kick: falling and drifting.')
    i = int(np.argmax([abs(t['in'][8]) for t in s]))
    add('Strongest drift', i, 'The drift sensor reads its largest value: a gust is pushing hard.')
    if tr['boulder']:
        d = [abs(t['in'][10]) + abs(t['in'][11]) for t in s]
        add('Closest to the boulder', int(np.argmin(d)), 'The boulder is nearest: the route sensors take over.')
    i = int(np.argmax([abs(t['in'][4]) for t in s]))
    add('Most tilted', i, 'The lander leans furthest: steering sideways, or fighting turbulence.')
    i = next((j for j, t in enumerate(s) if t['in'][6] or t['in'][7]), len(s) - 1)
    add('First touch', i, 'A leg touches the ground: from here it should only cushion the landing.')
    return out


def champions(run, config, seed_g):
    events = [json.loads(l) for l in open(os.path.join(ROOT, 'runs', run, 'events.jsonl')) if l.strip()]
    check = {e['champion']: e for e in events if e.get('champion')}
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'runs', run, 'champions', '*.pkl'))):
        rel = os.path.relpath(path, ROOT)
        m = re.match(r'g(\d+)_s(\d+)_(\w+)\.pkl', os.path.basename(path))
        g = pickle.load(open(path, 'rb'))
        e = check.get(rel, {})
        out.append({'file': os.path.basename(path), 'gen': int(m.group(1)), 'stage_index': int(m.group(2)),
                    'label': m.group(3), 'stage': e.get('stage', ''), 'check': e.get('check'),
                    'diff': diff(g, seed_g), 'net': export(g, config)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='D')
    a = ap.parse_args()
    config = load_config()
    seed_g = SeededGenome(0)
    mlp_to_genome(seed_g, config.genome_config, json.load(open(os.path.join(ROOT, 'models', 'seed_mlp15.json'))))
    champs = champions(a.run, config, seed_g)
    newest = pickle.load(open(os.path.join(ROOT, 'runs', a.run, 'champions', champs[-1]['file']), 'rb')) if champs else seed_g
    # the first of up to 10 analysis flights that lands (a landing is easier to learn from; the page says so)
    for i in range(10):
        tr = trace(newest, config, 3_000_555_001 + i)
        tr['tries'] = i + 1
        if tr['landed']:
            break
    cfg = {k: getattr(config.genome_config, k) for k in (
        'weight_mutate_rate', 'weight_mutate_power', 'weight_replace_rate', 'bias_mutate_rate', 'bias_mutate_power',
        'node_add_prob', 'node_delete_prob', 'conn_add_prob', 'conn_delete_prob', 'compatibility_disjoint_coefficient',
        'compatibility_weight_coefficient')}
    cfg.update(pop_size=config.pop_size, compatibility_threshold=config.species_set_config.compatibility_threshold,
               elitism=config.reproduction_config.elitism, survival_threshold=config.reproduction_config.survival_threshold,
               max_stagnation=config.stagnation_config.max_stagnation)
    data = {'run': a.run, 'input_names': INPUT_NAMES, 'output_names': OUT_NAMES, 'seed': export(seed_g, config),
            'champions': champs, 'newest': champs[-1]['file'] if champs else None, 'trace': tr,
            'presets': presets(tr), 'config': cfg}
    os.makedirs(os.path.join(ROOT, 'docs', 'data'), exist_ok=True)
    json.dump(data, open(os.path.join(ROOT, 'docs', 'data', 'network.json'), 'w'), separators=(',', ':'))
    print(f"network.json: {len(champs)} champions, flight {tr['outcome']} in {len(tr['steps'])} steps")


if __name__ == '__main__':
    main()
