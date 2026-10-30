"""How much does one NEAT mutation hurt a trained network? Children of the start network (one mutate() call each) on the
same real-task flights, for several weight-mutation settings.   python -m experiments.mutation_damage"""
import copy, pickle, random
from multiprocessing import Pool
import numpy as np
from lander.env import GOAL_WORLD, run_episode, make_env
from lander.neatnet import NeatPolicy
from make_network_data import load_config
from train import ANALYSIS
SETTINGS = [(0.4, 0.05), (0.2, 0.02), (0.4, 0.05, 'gentle'), (0.2, 0.02, 'gentle')]
GENTLE = dict(weight_replace_rate=0.0, bias_replace_rate=0.0, enabled_mutate_rate=0.0, conn_delete_prob=0.01,
              node_add_prob=0.01, conn_add_prob=0.05)
SEEDS = [ANALYSIS + 500_000 + i for i in range(150)]
def job(a):
    rate, power, gentle, child = a
    cfg = load_config(); gc = cfg.genome_config
    if gentle:
        for k, v in GENTLE.items(): setattr(gc, k, v)
    gc.weight_mutate_rate, gc.weight_mutate_power = rate, power
    gc.bias_mutate_rate, gc.bias_mutate_power = rate, power
    g = copy.deepcopy(pickle.load(open('models/seed16_genome.pkl', 'rb')))
    if child >= 0:
        random.seed(child); g.mutate(gc)
    pol, env = NeatPolicy(g, cfg), make_env()
    return np.mean([run_episode(pol, s, GOAL_WORLD, env=env)['landed'] for s in SEEDS])
if __name__ == '__main__':
    with Pool(8) as p:
        base = p.map(job, [(0, 0, None, -1)])[0]
        print(flush=True); print(f"start network: {base:.1%} on {len(SEEDS)} real-task flights")
        for st in SETTINGS:
            rate, power, gentle = (st + (None,))[:3]
            r = p.map(job, [(rate, power, gentle, c) for c in range(12)])
            print(f"rate {rate:4} power {power:5} {gentle or 'stock'}: children {np.mean(r):.1%} on average (worst {min(r):.0%}, best {max(r):.0%}), "
                  f"{sum(x < base - 0.05 for x in r)}/12 children more than 5 points worse")
