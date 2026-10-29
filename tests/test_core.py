import json, os, subprocess, sys

import neat
import numpy as np
import pytest

from lander.env import World, run_episode, make_env, boulder_spec, Flight
from lander.teacher import Teacher
from lander.clone import MLP
from lander.neatnet import mlp_to_genome, SeededGenome, NeatPolicy, SEED, size

ROOT = os.path.dirname(os.path.dirname(__file__))


def config():
    return neat.Config(SeededGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation,
                       os.path.join(ROOT, 'neat-config.ini'))


@pytest.mark.parametrize('world', [World(), World(boulder=True), World(steady=1.0, support=0.5)])
def test_same_seed_same_flight(world):
    env = make_env()
    a = run_episode(Teacher(), 123, world, env=env)
    run_episode(Teacher(), 999, World(boulder=True), env=env)        # reuse the env in between
    b = run_episode(Teacher(), 123, world, env=env)
    c = run_episode(Teacher(), 123, world)                           # and a brand-new env
    assert a['reward'] == b['reward'] == c['reward'] and a['steps'] == b['steps'] == c['steps']


def test_wind_depends_on_seed():
    assert run_episode(Teacher(), 1, World())['reward'] != run_episode(Teacher(), 2, World())['reward']


def test_boulder_hit_is_a_crash():
    # a policy that does nothing falls straight down; pick a seed whose boulder sits right on that line
    seed = next(s for s in range(10000) if abs(boulder_spec(s)[0] - 10) < 0.3)
    r = run_episode(lambda v, f: 0, seed, World(wind=0, turbulence=0, boulder=True))
    assert r['outcome'] == 'boulder' and not r['landed'] and r['reward'] < -50


def test_seed_ranges_do_not_overlap():
    from train import SEED_RANGES
    spans = sorted(SEED_RANGES.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_mlp_to_genome_flies_identically():
    mlp = MLP(13, 8, rng=np.random.default_rng(3))
    cfg = config()
    g = SeededGenome(1)
    mlp_to_genome(g, cfg.genome_config, mlp.to_dict())
    pol = NeatPolicy(g, cfg)
    X = np.random.default_rng(0).normal(0, 1, (200, 13))
    assert all(pol(x) == mlp(x) for x in X)
    assert size(g)[0] == 8                                          # at least one hidden layer, kept


def test_seed_hidden_nodes_are_never_deleted():
    SEED['net'], SEED['noise'] = MLP(13, 6).to_dict(), 0.0
    try:
        cfg = config()
        SeededGenome.protected = tuple(range(4, 10))
        g = SeededGenome(1)
        g.configure_new(cfg.genome_config)
        for _ in range(200):
            g.mutate_delete_node(cfg.genome_config)
        assert all(k in g.nodes for k in range(4, 10))
    finally:
        SEED['net'] = None


def test_smoke_train_then_resume(tmp_path):
    """Two generations, stop, resume: the generation counter and population carry on."""
    mlp = MLP(13, 4)
    seed = tmp_path / 'seed.json'
    seed.write_text(json.dumps(mlp.to_dict()))
    cur = tmp_path / 'c.json'
    cur.write_text(json.dumps({'name': 't', 'stages': [
        {'name': 'a', 'world': {'wind': 0, 'turbulence': 0}, 'cap': 100, 'pass': {'landed': 2}}]}))
    cfg = tmp_path / 'cfg.ini'
    cfg.write_text(open(os.path.join(ROOT, 'neat-config.ini')).read().replace('pop_size              = 120', 'pop_size = 8'))
    run = f'test_{os.getpid()}'
    base = [sys.executable, 'train.py', '--curriculum', str(cur), '--run', run, '--config', str(cfg),
            '--seed-net', str(seed), '--workers', '2', '--flights', '2', '--reeval-top', '2', '--reeval-flights', '2',
            '--check-every', '1', '--check-flights', '3']
    try:
        p = subprocess.Popen(base, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lines = []
        for line in p.stdout:
            lines.append(line)
            if sum(l.startswith('gen ') for l in lines) >= 3:
                p.terminate()
        p.wait(60)
        st = os.path.join(ROOT, 'runs', run, 'checkpoint.pkl')
        assert os.path.exists(st)
        import pickle
        gen = pickle.load(open(st, 'rb'))['state']['gen']
        assert gen >= 3
        p = subprocess.Popen(base, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        first = None
        for line in p.stdout:
            if line.startswith('gen '):
                first = int(line.split()[1])
                p.terminate()
                break
        p.wait(60)
        assert first == gen
    finally:
        import shutil
        shutil.rmtree(os.path.join(ROOT, 'runs', run), ignore_errors=True)
