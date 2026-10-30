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
    mlp = MLP(16, 8, rng=np.random.default_rng(3))
    mlp.W2 *= 20                                   # realistic: cloned output scores reach +-20 (tanh would tie them)
    cfg = config()
    g = SeededGenome(1)
    mlp_to_genome(g, cfg.genome_config, mlp.to_dict())
    pol = NeatPolicy(g, cfg)
    X = np.random.default_rng(0).normal(0, 1, (2000, 16))
    assert all(pol(x) == mlp(x) for x in X)
    assert size(g)[0] == 8                                          # at least one hidden layer, kept


def test_seed_hidden_nodes_are_never_deleted():
    SEED['net'], SEED['noise'] = MLP(16, 6).to_dict(), 0.0
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
    """Train, SIGTERM after 3 generations, resume: the generation counter carries on from the checkpoint."""
    import pickle, shutil, time
    seed = tmp_path / 'seed.json'
    seed.write_text(json.dumps(MLP(16, 4).to_dict()))
    cur = tmp_path / 'c.json'
    cur.write_text(json.dumps({'name': 't', 'stages': [
        {'name': 'a', 'world': {'wind': 0, 'turbulence': 0}, 'cap': 1000, 'pass': {'landed': 2}}]}))
    cfg = tmp_path / 'cfg.ini'
    cfg.write_text(open(os.path.join(ROOT, 'neat-config.ini')).read().replace('pop_size              = 120', 'pop_size = 8'))
    run = f'test_{os.getpid()}'
    rdir = os.path.join(ROOT, 'runs', run)
    cmd = [sys.executable, 'train.py', '--curriculum', str(cur), '--run', run, '--config', str(cfg),
           '--seed-net', str(seed), '--workers', '2', '--flights', '2', '--reeval-top', '2', '--reeval-flights', '2',
           '--check-every', '1', '--check-flights', '3']

    def rows():
        try:
            return open(os.path.join(rdir, 'stats.csv')).read().splitlines()[1:]
        except FileNotFoundError:
            return []

    def run_until(n_rows):
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            t0 = time.time()
            while len(rows()) < n_rows and time.time() - t0 < 600:   # generous: the CPU is often busy with training
                time.sleep(0.2)
            p.terminate()
            p.wait(600)
        finally:
            if p.poll() is None:
                p.kill()

    try:
        run_until(3)
        gen = pickle.load(open(os.path.join(rdir, 'checkpoint.pkl'), 'rb'))['state']['gen']
        assert gen >= 3
        before = len(rows())
        assert before == gen                       # one stats row per finished generation
        run_until(before + 1)
        assert int(rows()[before].split(',')[0]) == gen    # the resumed run's first generation
    finally:
        shutil.rmtree(rdir, ignore_errors=True)


def test_steady_push_follows_the_wind_ramp():
    env = make_env()
    assert Flight(env, 5, World(wind=5, steady=1.0)).push * 3 == pytest.approx(Flight(env, 5, World(wind=15, steady=1.0)).push)


def test_boulder_size_ramp_scales_only_the_radius():
    a, b = boulder_spec(42), boulder_spec(42, 0.4)
    assert a[:2] == b[:2] and b[2] == pytest.approx(0.4 * a[2])


def test_boulder_mix_and_check_world():
    from train import check_world
    env = make_env()
    n = sum(Flight(env, s, World(boulder_mix=0.3)).boulder is not None for s in range(300))
    assert 60 < n < 120
    assert check_world({'world': {'boulder_mix': 0.3}}).boulder_mix == 0.0


def test_goal_mix_rehearses_the_full_task():
    from train import check_world
    env = make_env()
    fl = [Flight(env, s, World(wind=0, turbulence=0, goal_mix=0.4)) for s in range(200)]
    real = [f for f in fl if f.boulder is not None]
    assert 50 < len(real) < 110 and all(f.world.wind == 15 and f.world.turbulence == 1.5 for f in real)
    assert check_world({'world': {'goal_mix': 0.4}}).goal_mix == 0.0


def test_boulder_top_stays_well_below_the_start():
    from lander.env import START_Y, PAD_Y
    for s in range(5000):
        x, y, r = boulder_spec(s)
        assert y + r <= START_Y - 3.5 + 1e-9 and y - PAD_Y >= 3.0 and 0.8 <= r <= 1.3


def test_resume_never_reuses_a_node_key(tmp_path):
    """A restored population must hand out new node keys above every genome's keys (run F crashed on this)."""
    import pickle, random as _r
    from types import SimpleNamespace
    import train
    cfg = config()
    pop = {}
    for i in range(6):
        g = SeededGenome(i); g.configure_new(cfg.genome_config)
        for _ in range(3 * i):
            g.mutate_add_node(cfg.genome_config)
        pop[i] = g
    cfg.genome_config.node_indexer = None
    run = SimpleNamespace(config=cfg, state=None)
    blob = {'population': pop, 'species': SimpleNamespace(species={}), 'generation': 5, 'random': _r.getstate(),
            'np': np.random.get_state(), 'state': {'gen': 5, 'stage': 0}}
    p = tmp_path / 'ck.pkl'; pickle.dump(blob, open(p, 'wb'))
    run.cur = {'stages': [{'name': 'a'}]}
    run.stage = lambda: run.cur['stages'][0]
    import neat as _neat
    orig = _neat.Population
    _neat.Population = lambda c, st: SimpleNamespace()
    try:
        train.Run._restore(run, str(p))
    finally:
        _neat.Population = orig
    new = next(cfg.genome_config.node_indexer)
    assert all(new > max(g.nodes) for g in pop.values())


def test_fastnet_matches_neat_python():
    """The compiled network gives the same outputs as neat-python's, incl. evolved extra nodes and direct connections."""
    import random as _r
    cfg = config()
    _r.seed(4)
    g = SeededGenome(1)
    mlp = MLP(16, 6, rng=np.random.default_rng(1)); mlp.W2 *= 20
    mlp_to_genome(g, cfg.genome_config, mlp.to_dict())
    for _ in range(6):
        g.mutate_add_node(cfg.genome_config)
        g.mutate_add_connection(cfg.genome_config)
    slow, fast = NeatPolicy(g, cfg, fast=False), NeatPolicy(g, cfg)
    X = np.random.default_rng(2).normal(0, 1, (3000, 16))
    for x in X:
        a, b = slow.net.activate(list(x)), fast.net.activate(x)
        assert np.allclose(a, b, atol=1e-9)
    env = make_env()
    r1 = run_episode(slow, 77, World(boulder=True), env=env)
    r2 = run_episode(fast, 77, World(boulder=True), env=env)
    assert r1['reward'] == pytest.approx(r2['reward']) and r1['steps'] == r2['steps']
