"""NEAT through a curriculum, starting from a cloned network with a hidden layer.

    python train.py --curriculum curricula/A.json --run A --workers 6          # new run (or resume if it exists)

The curriculum file lists stages; each stage has a world (lander/env.py World), optional ramps
(e.g. wind 5 -> 15 over the first part of the stage), training-wheel support that fades out, a reward,
a pass check and a generation cap. The population carries on from stage to stage.

Every generation: all genomes fly the same K training seeds (common random numbers). The top few are re-flown
on more selection seeds and the best of those is the generation champion. Every `check_every` generations
the champion flies 100 fresh check seeds in the stage's final world with no support; if it passes, a second
fresh 100 confirms it (the winner's curse) and the stage ends.

SIGTERM: finish the generation, checkpoint, exit. Resume: the same command.
"""

import argparse
import csv
import json
import os
import pickle
import random
import signal
import time
from itertools import count
from multiprocessing import Pool

import neat
import numpy as np

from lander.env import World, GOAL_WORLD, run_episode, make_env, LANDING_BONUS, centre_bonus
from lander.neatnet import SEED, SeededGenome, NeatPolicy, size

# Seed ranges: never mix them up (tests/test_seeds.py checks they don't overlap)
TRAIN = 0                    # + global_gen * 1000 + i
CLONE = 500_000_000          # lander/clone.py
CHECK = 1_000_000_000        # + n_check * 1000 + i : stage pass checks, each block used once
SELECT = 1_500_000_000       # + global_gen * 1000 + i : re-flying the top genomes
TEST = 2_000_000_000         # final goal check (experiments/goal_check.py)
ANALYSIS = 3_000_000_000     # comparisons, failure analysis
SEED_RANGES = {'train': (TRAIN, CLONE), 'clone': (CLONE, CHECK), 'check': (CHECK, SELECT),
               'select': (SELECT, TEST), 'test': (TEST, ANALYSIS), 'analysis': (ANALYSIS, 4_000_000_000)}

_W = {}


def _init_worker(config):
    _W['config'] = config
    _W['env'] = make_env()


def fly_genome(args):
    """Fly one genome on the given seeds; return the per-flight results (small dicts)."""
    genome, seeds, world, reward = args
    pol = NeatPolicy(genome, _W['config'])
    return [shape(run_episode(pol, s, world, env=_W['env']), reward) for s in seeds]


def fly_many(pool, jobs, chunk=10):
    """Fly [(genome, seeds, world, reward), ...] with every job split into chunks of `chunk` seeds, so all workers stay
    busy even when there are fewer jobs than workers (8 genomes re-flown on 60 seeds = 48 tasks, not 8).
    Returns one result list per job, in order."""
    tasks, owner = [], []
    for j, (g, seeds, world, reward) in enumerate(jobs):
        for k in range(0, len(seeds), chunk):
            tasks.append((g, seeds[k:k + chunk], world, reward))
            owner.append(j)
    out = [[] for _ in jobs]
    for j, rs in zip(owner, pool.map(fly_genome, tasks, chunksize=1)):
        out[j] += rs
    return out


def shape(r, reward):
    """Training fitness for one flight: env score + landing bonus + centre bonus, a timeout penalty,
    an extra boulder-hit penalty and a small clearance bonus (all set per stage)."""
    f = r['reward']
    if r['landed']:
        f += reward.get('landing_bonus', LANDING_BONUS) + reward.get('centre', 1.0) * centre_bonus(r['x_final'])
    if r['outcome'] == 'timeout':
        f -= reward.get('timeout_penalty', 100.0)
    if r['outcome'] == 'boulder':
        f -= reward.get('boulder_penalty', 0.0)
    if r['clearance'] is not None and reward.get('clearance_bonus'):
        f += reward['clearance_bonus'] * min(max(r['clearance'], 0.0), 1.0)
    return {'f': f, 'reward': r['reward'], 'landed': r['landed'], 'outcome': r['outcome']}


def summarise(rs):
    return {'fitness': float(np.mean([r['f'] for r in rs])), 'score': float(np.mean([r['reward'] for r in rs])),
            'landed': float(np.mean([r['landed'] for r in rs])),
            'crashed': float(np.mean([r['outcome'] == 'crashed' for r in rs])),
            'boulder': float(np.mean([r['outcome'] == 'boulder' for r in rs])),
            'timeout': float(np.mean([r['outcome'] == 'timeout' for r in rs]))}


def stage_world(stage, t):
    """The stage's world at progress t (0..1 of its ramp). t >= 1: the final world, support off."""
    w = dict(stage.get('world', {}))
    for k, (a, b) in stage.get('ramp', {}).items():
        w[k] = a + (b - a) * min(t, 1.0)
    s0, s1 = stage.get('support', [0.0, 0.0])
    w['support'] = 0.0 if t >= 1 else s0 + (s1 - s0) * t
    return World(**w)


def check_world(stage):
    """The world a pass check flies: the final stage world, no support, no mixed-in boulders."""
    w = stage_world(stage, 1.0)
    w.boulder_mix = w.goal_mix = 0.0
    return w


class Run:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join('runs', a.run)
        os.makedirs(os.path.join(self.dir, 'champions'), exist_ok=True)
        self.cur = json.load(open(a.curriculum))
        json.dump(self.cur, open(os.path.join(self.dir, 'curriculum.json'), 'w'), indent=1)
        self.config = neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                  neat.DefaultStagnation, a.config)
        SEED['net'] = json.load(open(a.seed_net)) if a.seed_net else None
        SEED['noise'] = a.seed_noise
        if SEED['net']:
            H = len(SEED['net']['b1'])
            SeededGenome.protected = tuple(range(4, 4 + H))
        self.stop = False
        signal.signal(signal.SIGTERM, self._sigterm)
        signal.signal(signal.SIGINT, self._sigterm)
        ck = os.path.join(self.dir, 'checkpoint.pkl')
        if os.path.exists(ck):
            self._restore(ck)
        else:
            self.pop = neat.Population(self.config)
            self.state = {'gen': 0, 'stage': 0, 'stage_gen': 0, 'n_check': 0, 'best_check': {}, 'done': False}
            self._event('run_start', note=self.cur.get('note', ''), args=vars(a))
            self._event('stage_start', stage=self.stage()['name'], expect=self.stage().get('expect', ''))
        self.pop.add_reporter(neat.StdOutReporter(False)) if a.verbose else None

    def _sigterm(self, *_):
        print('signal: checkpoint after this generation', flush=True)
        self.stop = True

    def stage(self):
        return self.cur['stages'][self.state['stage']]

    # --- persistence ---
    def _save(self):
        p = self.pop
        blob = {'state': self.state, 'population': p.population, 'species': p.species, 'generation': p.generation,
                'random': random.getstate(), 'np': np.random.get_state()}
        tmp = os.path.join(self.dir, 'checkpoint.pkl.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(blob, f)
        os.replace(tmp, os.path.join(self.dir, 'checkpoint.pkl'))

    def _restore(self, path):
        blob = pickle.load(open(path, 'rb'))
        self.pop = neat.Population(self.config, (blob['population'], blob['species'], blob['generation']))
        # new hidden-node keys must not collide with ANY genome's keys. neat-python would otherwise restart its counter
        # from the first genome it mutates, and crash on an assert when another genome already has that key (run F, 21:14)
        top = max(max(g.nodes) for g in blob['population'].values())
        top = max([top] + [max(s.representative.nodes) for s in blob['species'].species.values()])
        self.config.genome_config.node_indexer = count(top + 1)
        random.setstate(blob['random'])
        np.random.set_state(blob['np'])
        self.state = blob['state']
        print(f"resumed at gen {self.state['gen']} stage {self.stage()['name']}", flush=True)

    def _event(self, kind, **kw):
        e = {'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'kind': kind, 'gen': self.state['gen'],
             'stage_index': self.state['stage'], **kw}
        with open(os.path.join(self.dir, 'events.jsonl'), 'a') as f:
            f.write(json.dumps(e) + '\n')

    def _save_champion(self, genome, label):
        st = self.state
        path = os.path.join(self.dir, 'champions', f"g{st['gen']:05d}_s{st['stage']}_{label}.pkl")
        pickle.dump(genome, open(path, 'wb'))
        return path

    # --- one generation ---
    def evaluate(self, genomes, config):
        st, stg, a = self.state, self.stage(), self.a
        cap = stg['cap']
        ramp_len = max(1, int(stg.get('ramp_frac', 0.6) * cap))
        t = st['stage_gen'] / ramp_len if (stg.get('ramp') or stg.get('support')) else 1.0
        world = stage_world(stg, t)
        reward = stg.get('reward', {})
        K = stg.get('flights', a.flights)
        seeds = [TRAIN + st['gen'] * 1000 + i for i in range(K)]
        t0 = time.time()
        res = self.pool.map(fly_genome, [(g, seeds, world, reward) for _, g in genomes], chunksize=2)
        for (gid, g), rs in zip(genomes, res):
            g.fitness = summarise(rs)['fitness']
        # re-fly the top genomes on fresh selection seeds; champion = best average over all its flights
        top = sorted(zip(genomes, res), key=lambda z: -z[0][1].fitness)[:a.reeval_top]
        sel_seeds = [SELECT + st['gen'] * 1000 + i for i in range(a.reeval_flights)]
        more = fly_many(self.pool, [(g, sel_seeds, world, reward) for (_, g), _ in top])
        scored = [(summarise(rs + m), g) for ((_, g), rs), m in zip(top, more)]
        champ_s, champ = max(scored, key=lambda z: z[0]['fitness'])
        fits = [g.fitness for _, g in genomes]
        row = {'gen': st['gen'], 'stage': stg['name'], 'stage_gen': st['stage_gen'], 'wind': world.wind,
               'turbulence': world.turbulence, 'steady': world.steady, 'boulder': int(world.boulder),
               'support': round(world.support, 3), 'best': round(max(fits), 1), 'mean': round(float(np.mean(fits)), 1),
               'champ_fit': round(champ_s['fitness'], 1), 'champ_landed': round(champ_s['landed'], 3),
               'champ_score': round(champ_s['score'], 1), 'champ_timeout': round(champ_s['timeout'], 3),
               'champ_boulder': round(champ_s['boulder'], 3),
               'hidden': size(champ)[0], 'conns': size(champ)[1], 'species': len(self.pop.species.species),
               'check_landed': '', 'check_score': '', 'secs': 0}
        # the real task, tracked every check_every generations whatever the stage (monitoring only, not selection)
        if st['stage_gen'] % a.check_every == 0:
            g = self.check(champ, GOAL_WORLD, reward)
            self._goal_track(champ, g)
        # pass check: only once the ramp is done (final world, no support)
        if t >= 1 and st['stage_gen'] % a.check_every == 0:
            final = check_world(stg)
            chk = self.check(champ, final, reward)
            row['check_landed'], row['check_score'] = round(chk['landed'], 3), round(chk['score'], 1)
            best = st['best_check'].get(str(st['stage']), -1)
            key = chk['landed'] + chk['score'] / 1e4
            if key > best:
                st['best_check'][str(st['stage'])] = key
                path = self._save_champion(champ, 'best')
                self._event('new_best', stage=stg['name'], check=chk, champion=path)
            if self.passes(chk, stg['pass']):
                conf = self.check(champ, final, reward)
                self._event('check_pass', stage=stg['name'], check=chk, confirm=conf)
                if self.passes(conf, stg['pass']):
                    path = self._save_champion(champ, 'passed')
                    self._event('stage_pass', stage=stg['name'], check=chk, confirm=conf, champion=path)
                    self._next_stage(passed=True)
        row['secs'] = round(time.time() - t0, 1)
        self._log(row)
        self.champion = champ

    def _goal_track(self, champ, g):
        st = self.state
        path = os.path.join(self.dir, 'goal_track.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)                       # stage names contain commas: let csv quote them
            if new:
                w.writerow(['gen', 'stage', 'landed', 'score', 'crashed', 'boulder', 'timeout'])
            w.writerow([st['gen'], self.stage()['name'], f"{g['landed']:.3f}", f"{g['score']:.1f}", f"{g['crashed']:.3f}",
                        f"{g['boulder']:.3f}", f"{g['timeout']:.3f}"])
        best = st.setdefault('best_goal', -1)
        if g['landed'] > best:
            st['best_goal'] = g['landed']
            path = self._save_champion(champ, 'goal')
            self._event('goal_best', stage=self.stage()['name'], check=g, champion=path)

    def check(self, genome, world, reward):
        st = self.state
        seeds = [CHECK + st['n_check'] * 1000 + i for i in range(self.a.check_flights)]
        st['n_check'] += 1
        return summarise(fly_many(self.pool, [(genome, seeds, world, reward)], chunk=5)[0])

    @staticmethod
    def passes(chk, p):
        return (chk['landed'] >= p.get('landed', 0) and chk['score'] >= p.get('score', -1e9)
                and chk['boulder'] <= p.get('boulder', 1.0) and chk['timeout'] <= p.get('timeout', 1.0))

    def _next_stage(self, passed):
        st = self.state
        if not passed:
            self._event('stage_cap', stage=self.stage()['name'])
        if st['stage'] + 1 >= len(self.cur['stages']):
            st['done'] = True
            self._event('curriculum_done', passed=passed)
            return
        st['stage'] += 1
        st['stage_gen'] = -1           # incremented to 0 after this generation
        self._event('stage_start', stage=self.stage()['name'], expect=self.stage().get('expect', ''))

    def _log(self, row):
        path = os.path.join(self.dir, 'stats.csv')
        new = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(' | '.join(f'{k} {v}' for k, v in row.items() if v != ''), flush=True)

    def _overrides(self):
        path = os.path.join(self.dir, 'override.json')
        if os.path.exists(path):
            o = json.load(open(path))
            os.remove(path)
            if o.get('next_stage'):
                self._event('override', what='next_stage', reason=o.get('reason', ''))
                self._next_stage(passed=False)
                self.state['stage_gen'] += 1

    def train(self):
        with Pool(self.a.workers, initializer=_init_worker, initargs=(self.config,)) as self.pool:
            while not self.state['done'] and not self.stop:
                self.pop.run(self.evaluate, 1)
                st = self.state
                st['gen'] += 1
                st['stage_gen'] += 1
                if st['stage_gen'] >= self.stage()['cap'] and not st['done']:
                    self._next_stage(passed=False)
                    st['stage_gen'] += 1
                self._overrides()
                if st['gen'] % 5 == 0 or self.stop or st['done']:
                    self._save()
        self._save()
        print('stopped' if self.stop else 'curriculum done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curriculum', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--config', default='neat-config.ini')
    ap.add_argument('--seed-net', default='models/seed_mlp15.json')
    ap.add_argument('--seed-noise', type=float, default=0.02)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--flights', type=int, default=10)
    ap.add_argument('--reeval-top', type=int, default=8)
    ap.add_argument('--reeval-flights', type=int, default=40)
    ap.add_argument('--check-every', type=int, default=5)
    ap.add_argument('--check-flights', type=int, default=100)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    Run(a).train()


if __name__ == '__main__':
    main()
