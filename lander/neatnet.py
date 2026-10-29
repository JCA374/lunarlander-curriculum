"""NEAT networks that start from a good network with a hidden layer.

- SeededGenome: neat-python's DefaultGenome whose first population is copies of one seed network (an MLP
  cloned from the teacher, see lander/clone.py), with a little weight noise. The seed's hidden nodes are
  protected: never deleted (run K in the wind project lost its 4 hidden nodes within 400 generations).
  Everything else evolves as usual: weights, biases, new nodes, new connections.
- mlp_to_genome / NeatPolicy: conversion and the policy wrapper used by run_episode.
"""

from itertools import count
from random import choice

import neat
import numpy as np

TANH_GAIN = 2.5
SEED = {'net': None, 'noise': 0.0}     # set by train.py before the population is created


def mlp_to_genome(genome, config, mlp):
    """Fill an empty genome with an MLP: mlp = {'W1': (H, n_in), 'b1': (H,), 'W2': (n_out, H), 'b2': (n_out,)}.
    NEAT keys: inputs -1..-n_in, outputs 0..n_out-1, hidden n_out..n_out+H-1."""
    gc = config
    W1, b1, W2, b2 = (np.asarray(mlp[k]) for k in ('W1', 'b1', 'W2', 'b2'))
    W1, b1 = W1 / TANH_GAIN, b1 / TANH_GAIN     # neat-python's tanh is tanh(2.5 z)
    n_out, H = W2.shape
    hidden = list(range(n_out, n_out + H))
    genome.nodes, genome.connections = {}, {}
    for o in range(n_out):
        n = genome.create_node(gc, o)
        # identity, not tanh: the cloned output scores reach +-20, and tanh(2.5 z) rounds several of them to
        # exactly 1.0, so argmax picked the first of a tie, not the best action (fixed 2026-10-29, attempt D)
        n.bias, n.activation, n.response = float(b2[o]), 'identity', 1.0
        genome.nodes[o] = n
    for j, k in enumerate(hidden):
        n = genome.create_node(gc, k)
        n.bias, n.activation, n.response = float(b1[j]), 'tanh', 1.0
        genome.nodes[k] = n
        for i, ik in enumerate(gc.input_keys):
            c = genome.create_connection(gc, ik, k)
            c.weight, c.enabled = float(W1[j, i]), True
            genome.connections[(ik, k)] = c
        for o in range(n_out):
            c = genome.create_connection(gc, k, o)
            c.weight, c.enabled = float(W2[o, j]), True
            genome.connections[(k, o)] = c
    if gc.node_indexer is None:
        gc.node_indexer = count(hidden[-1] + 1)     # new hidden nodes never reuse the seed's keys
    return hidden


class SeededGenome(neat.DefaultGenome):
    protected = ()

    def configure_new(self, config):
        if SEED['net'] is None:
            return super().configure_new(config)
        hidden = mlp_to_genome(self, config, SEED['net'])
        SeededGenome.protected = tuple(hidden)
        s = SEED['noise']
        if s:
            for c in self.connections.values():
                c.weight += float(np.random.normal(0, s))
            for n in self.nodes.values():
                n.bias += float(np.random.normal(0, s))

    def mutate_delete_node(self, config):
        prot = set(config.output_keys) | set(SeededGenome.protected)
        available = [k for k in self.nodes if k not in prot]
        if not available:
            return -1
        k = choice(available)
        for key in [c for c in self.connections if k in c]:
            del self.connections[key]
        del self.nodes[k]
        return k


class NeatPolicy:
    def __init__(self, genome, config):
        self.net = neat.nn.FeedForwardNetwork.create(genome, config)
        self.n = len(config.genome_config.input_keys)

    def __call__(self, v, flight=None):
        return int(np.argmax(self.net.activate(v[:self.n])))   # a 13-input network ignores the route sensors


def size(genome):
    hidden = [k for k in genome.nodes if k >= 4]
    return len(hidden), sum(c.enabled for c in genome.connections.values())
