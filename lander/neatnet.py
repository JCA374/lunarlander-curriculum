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


class FastNet:
    """A NEAT feed-forward network compiled to a few NumPy matrix products (one per layer), about 10x faster than
    neat-python's activate(), which loops over every connection in Python (70% of all training time, profiled
    2026-10-30). Same layers, weights, biases, responses and activations as neat-python; only the order of float
    additions differs, so results match to ~1e-12."""

    def __init__(self, genome, config):
        from neat.graphs import feed_forward_layers
        gc = config.genome_config
        self.inputs, self.outputs = list(gc.input_keys), list(gc.output_keys)
        conns = [(k, c.weight) for k, c in genome.connections.items() if c.enabled]
        layers = feed_forward_layers(self.inputs, self.outputs, [k for k, _ in conns])
        order = self.inputs + [n for layer in layers for n in layer]
        idx = {k: i for i, k in enumerate(order)}
        for o in self.outputs:                         # an output nobody feeds still exists (its value: act(bias))
            if o not in idx:
                idx[o] = len(order); order.append(o); layers.append({o})
        self.size, self.idx = len(order), idx
        self.layers = []
        for layer in layers:
            keys = sorted(layer)
            W = np.zeros((len(keys), self.size))
            for (a, b), w in conns:
                if b in layer and a in idx:
                    W[keys.index(b), idx[a]] = w
            nodes = [genome.nodes[k] for k in keys]
            resp = np.array([n.response for n in nodes])
            bias = np.array([n.bias for n in nodes])
            tanh = np.array([n.activation == 'tanh' for n in nodes])
            if not all(n.activation in ('tanh', 'identity') for n in nodes):
                raise ValueError('FastNet supports tanh and identity nodes only')
            rows = np.array([idx[k] for k in keys])
            W = W * resp[:, None]
            # tanh rows get the 2.5 gain folded in (neat-python: tanh(2.5 z), clipped at +-60: tanh is already +-1 there)
            t, i = tanh, ~tanh
            self.layers.append((rows[t], 2.5 * W[t], 2.5 * bias[t], rows[i], W[i], bias[i]))
        self.out_idx = np.array([idx[o] for o in self.outputs])
        self.n_in = len(self.inputs)
        self.values = {}

    def activate(self, inputs):
        v = self.buf if hasattr(self, 'buf') else np.zeros(self.size)
        self.buf = v
        v[:self.n_in] = inputs
        for tr, tW, tb, ir, iW, ib in self.layers:
            if len(tr):
                v[tr] = np.tanh(tW @ v + tb)
            if len(ir):
                v[ir] = iW @ v + ib
        self.v = v
        return v[self.out_idx]

    @property
    def node_values(self):
        return {k: float(self.v[i]) for k, i in self.idx.items() if k >= 0}


class NeatPolicy:
    def __init__(self, genome, config, fast=True):
        self.net = FastNet(genome, config) if fast else neat.nn.FeedForwardNetwork.create(genome, config)
        self.n = len(config.genome_config.input_keys)

    def __call__(self, v, flight=None):
        return int(np.argmax(self.net.activate(v[:self.n])))   # a 13-input network ignores the route sensors

    @property
    def values(self):
        return self.net.node_values if isinstance(self.net, FastNet) else self.net.values


def size(genome):
    hidden = [k for k in genome.nodes if k >= 4]
    return len(hidden), sum(c.enabled for c in genome.connections.values())
