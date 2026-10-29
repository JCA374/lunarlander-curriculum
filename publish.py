"""Build the site's data from the runs and push it (run every 15 min by publish_loop.sh).

docs/data/status.json   run, stage, latest numbers, goal table
docs/data/stats.json    per-generation numbers for the charts (every run)
docs/data/log.json      newest-first log: journal notes + run events, each event with a replay GIF of the
                        champion *as it was then* (rendered once, when the event is first seen)
docs/data/curricula.json  every curriculum (stages, rewards, pass checks, expected outcomes)

    python publish.py            # build only
    python publish.py --push     # build, commit docs/ + run logs, push
"""

import argparse
import csv
import glob
import json
import os
import pickle
import subprocess
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, 'docs')
DATA = os.path.join(DOCS, 'data')
REPLAYS = os.path.join(DOCS, 'replays')
DEADLINE = '2026-11-04 18:00'
REPLAY_KINDS = {'new_best', 'stage_pass', 'goal_check'}


def read_stats(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_events(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def replay_gif(champion_path, world, seed, out):
    """Render one flight of a saved champion as a small GIF (every 2nd frame, half size), boulder drawn."""
    import neat
    from PIL import Image, ImageDraw
    from lander.env import run_episode, make_env, SCALE
    from lander.neatnet import NeatPolicy, SeededGenome
    cfg = neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                      os.path.join(ROOT, 'neat-config.ini'))
    genome = pickle.load(open(champion_path, 'rb'))
    env = make_env('rgb_array')
    frames = []

    def rec(f):
        if f.steps % 2:
            return
        img = Image.fromarray(env.render())
        if f.boulder:
            bx, by, r = f.boulder
            d = ImageDraw.Draw(img)
            cx, cy, rr = bx * SCALE, 400 - by * SCALE, r * SCALE
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(130, 115, 100), outline=(200, 190, 170), width=2)
        frames.append(img.resize((300, 200)).convert('P', palette=Image.ADAPTIVE, colors=32))

    r = run_episode(NeatPolicy(genome, cfg), seed, world, env=env, record=rec)
    env.close()
    frames += [frames[-1]] * 15
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
    return r


def build():
    from train import stage_world, ANALYSIS
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(REPLAYS, exist_ok=True)
    log = []
    stats_all, curricula, runs = {}, {}, []
    for rdir in sorted(glob.glob(os.path.join(ROOT, 'runs', '*'))):
        name = os.path.basename(rdir)
        if name.startswith('test_') or not os.path.exists(os.path.join(rdir, 'stats.csv')):
            continue
        cur = json.load(open(os.path.join(rdir, 'curriculum.json')))
        curricula[name] = cur
        rows = read_stats(os.path.join(rdir, 'stats.csv'))
        keep = ['gen', 'stage', 'best', 'mean', 'champ_fit', 'champ_landed', 'champ_score', 'check_landed',
                'check_score', 'wind', 'turbulence', 'support', 'boulder', 'hidden', 'conns', 'species', 'secs']
        stats_all[name] = [{k: r.get(k, '') for k in keep} for r in rows]
        events = read_events(os.path.join(rdir, 'events.jsonl'))
        for i, e in enumerate(events):
            e = dict(e, run=name)
            if e['kind'] in REPLAY_KINDS and e.get('champion'):
                gif = f"{name}_{i:04d}.gif"
                path = os.path.join(REPLAYS, gif)
                if not os.path.exists(path) and os.path.exists(os.path.join(ROOT, e['champion'].lstrip('./'))):
                    stage = cur['stages'][e['stage_index']]
                    try:
                        replay_gif(os.path.join(ROOT, e['champion']), stage_world(stage, 1.0),
                                   ANALYSIS + 777_000 + i, path)
                    except Exception as ex:      # a replay must never stop the publisher
                        print('replay failed', ex)
                if os.path.exists(path):
                    e['replay'] = f'replays/{gif}'
            e.pop('args', None)
            log.append(e)
        last = rows[-1] if rows else {}
        alive = subprocess.run(['pgrep', '-f', f'[t]rain.py .*--run {name}'], capture_output=True).returncode == 0
        runs.append({'run': name, 'note': cur.get('note', ''), 'alive': alive, 'last': last,
                     'n_stages': len(cur['stages']),
                     'passed': [e['stage'] for e in events if e['kind'] == 'stage_pass'],
                     'best_check': max((e['check'] for e in events if e['kind'] == 'new_best' and
                                        e['stage_index'] == len(cur['stages']) - 1), default=None,
                                       key=lambda c: c['landed'])})
    jpath = os.path.join(ROOT, 'experiments', 'journal.jsonl')
    for e in read_events(jpath):
        log.append(dict(e, kind='journal'))
    for g in sorted(glob.glob(os.path.join(ROOT, 'results', 'goal_*.json'))):
        r = json.load(open(g))
        log.append({'t': r['t'], 'kind': 'goal_check', **r})
    log.sort(key=lambda e: e['t'], reverse=True)
    status = {'updated': time.strftime('%Y-%m-%d %H:%M'), 'deadline': DEADLINE, 'goal_landed': 0.98, 'runs': runs}
    for name, obj in [('status', status), ('stats', stats_all), ('log', log), ('curricula', curricula)]:
        json.dump(obj, open(os.path.join(DATA, f'{name}.json'), 'w'), separators=(',', ':'))


def push():
    os.chdir(ROOT)
    subprocess.run(['git', 'add', 'docs', 'experiments/journal.jsonl', 'results'], check=False)
    for f in glob.glob('runs/*/stats.csv') + glob.glob('runs/*/events.jsonl') + glob.glob('runs/*/curriculum.json') \
            + glob.glob('runs/*/champions/*.pkl'):
        if '/test_' not in f:
            subprocess.run(['git', 'add', '-f', f])
    if subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 0:
        return print('nothing to publish')
    subprocess.run(['git', 'commit', '-q', '-m', f'Site update {time.strftime("%Y-%m-%d %H:%M")}'], check=True)
    subprocess.run(['git', 'pull', '-q', '--rebase'], check=False)
    subprocess.run(['git', 'push', '-q'], check=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--push', action='store_true')
    a = ap.parse_args()
    build()
    if a.push:
        push()
