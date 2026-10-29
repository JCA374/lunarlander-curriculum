"""Flying one LunarLander episode in wind, turbulence and (optionally) past a boulder.

Ported from ../lunarlander-neat-wind/neat_lander/env.py (seed_wind, steady_push, DriftSensor), plus:
- World: what the flight looks like (wind, turbulence, steady push, boulder, training-wheel support).
- the boulder: a static circle in the air between the start and the pad. Touching it ends the flight as a crash.
- a fixed 13-number input vector for every policy (see inputs()).
"""

import math
from dataclasses import dataclass, asdict

import gymnasium as gym
import numpy as np
from Box2D.b2 import circleShape

ENV_ID = 'LunarLander-v2'
MAX_STEPS = 1000
WIND_POWER = 15.0
TURBULENCE_POWER = 1.5
SCALE = 30.0
W, H = 600 / SCALE, 400 / SCALE          # world size in metres (20 x 13.3)
PAD_Y = H / 4                             # helipad height in metres
LEG_DOWN = 18 / SCALE
PAD_HALF_WIDTH = 0.2                      # flags at x = +-0.2 in observation units
CENTRE_BONUS = 50.0
LANDING_BONUS = 100.0
N_INPUTS = 13
INPUT_NAMES = ['x', 'y', 'vx', 'vy', 'angle', 'vangle', 'leg_l', 'leg_r', 'drift', '|x|',
               'boulder_dx', 'boulder_dy', 'boulder_r']


@dataclass
class World:
    wind: float = WIND_POWER              # gymnasium wind_power (0 = calm)
    turbulence: float = TURBULENCE_POWER  # gymnasium turbulence_power
    steady: float = 0.0                   # 0..1: share of the wind replaced by one constant push (curriculum)
    boulder: bool = False                 # a boulder on the approach
    support: float = 0.0                  # training wheels: a stabilising torque, 1 = full, 0 = off

    def as_dict(self):
        return asdict(self)


GOAL_WORLD = World(boulder=True)


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode, enable_wind=True, wind_power=WIND_POWER,
                    turbulence_power=TURBULENCE_POWER)


def seed_wind(env, seed):
    """Gymnasium draws the gust phase from the global numpy RNG, not from the reset seed. Tie it to the
    seed, so one seed means the same wind for every policy, every time."""
    rng = np.random.default_rng([int(seed), 7])
    env.unwrapped.wind_idx = int(rng.integers(-9999, 9999))
    env.unwrapped.torque_idx = int(rng.integers(-9999, 9999))


def steady_push(seed):
    """The constant sideways force of a curriculum flight: 30-100% of WIND_POWER, either direction."""
    rng = np.random.default_rng([int(seed), 11])
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * WIND_POWER)


def boulder_spec(seed):
    """Where the boulder is for this seed: centre (metres) and radius. It floats on the approach,
    1.6-2.6 m wide, its centre 3-7 m above the pad and up to 2.5 m to either side of the middle, so it
    often sits right on the straight line down."""
    rng = np.random.default_rng([int(seed), 13])
    r = float(rng.uniform(0.8, 1.3))
    x = float(W / 2 + rng.uniform(-2.5, 2.5))
    y = float(PAD_Y + rng.uniform(3.0, 7.0))
    return x, y, r


# Drift sensor: the wind shows in the observations as a change in the horizontal velocity. The sensor
# averages that change over the recent steps, leaving out side-engine steps: an accelerometer.
DRIFT_SCALE = 1 / (0.0008 * WIND_POWER)
DRIFT_RATE = 0.15


class DriftSensor:
    def __init__(self):
        self.value, self.prev_vx = 0.0, None

    def update(self, obs, last_action):
        touching = obs[6] or obs[7]
        if self.prev_vx is not None and last_action in (0, 2) and not touching:
            self.value += DRIFT_RATE * ((obs[2] - self.prev_vx) * DRIFT_SCALE - self.value)
        self.prev_vx = obs[2]
        return self.value


def lander_xy(env):
    p = env.unwrapped.lander.position
    return float(p.x), float(p.y)


def inputs(obs, drift, boulder, lander_pos):
    """The 13 numbers every policy sees: 8 observations, the drift sensor, |x|, and the boulder's
    position relative to the lander and its radius, in observation units (x / 10 m, y / 6.67 m).
    Without a boulder the last three are 0."""
    v = list(map(float, obs)) + [drift, abs(float(obs[0]))]
    if boulder is None:
        v += [0.0, 0.0, 0.0]
    else:
        bx, by, r = boulder
        lx, ly = lander_pos
        v += [(bx - lx) / (W / 2), (by - ly) / (H / 2), r / (W / 2)]
    return np.array(v)


class Flight:
    """One flight: env + world + seed. step(action) -> done. Used by run_episode and the hand controller."""

    def __init__(self, env, seed, world=World()):
        self.env, self.seed, self.world = env, int(seed), world
        u = env.unwrapped
        seed_wind(env, seed)
        u.wind_power = world.wind * (1 - world.steady)
        u.turbulence_power = world.turbulence * (1 - world.steady)
        u.enable_wind = (world.wind > 0 or world.turbulence > 0) and world.steady < 1
        # the push scales with the stage's wind, so a wind ramp also ramps the steady push
        self.push = world.steady * steady_push(seed) * world.wind / WIND_POWER if world.steady else 0.0
        _remove_boulder(env)
        self.obs, _ = env.reset(seed=self.seed)
        self.boulder = boulder_spec(seed) if world.boulder else None
        if self.boulder:
            _add_boulder(env, *self.boulder)
        self.sensor = DriftSensor()
        self.drift = 0.0
        self.action = 0
        self.reward = 0.0          # env score
        self.steps = 0
        self.outcome = None
        self.min_clearance = math.inf
        self.touch_speed = None

    def inputs(self):
        return inputs(self.obs, self.drift, self.boulder, lander_xy(self.env))

    def touching(self):
        u = self.env.unwrapped
        return u.legs[0].ground_contact or u.legs[1].ground_contact

    def step(self, action):
        u = self.env.unwrapped
        if self.push and not self.touching():
            u.lander.ApplyForceToCenter((self.push, 0.0), True)
        if self.world.support:
            # training wheels: a torque that pulls the lander upright, fading out across the curriculum
            t = -self.world.support * (8.0 * u.lander.angle + 2.0 * u.lander.angularVelocity)
            u.lander.ApplyTorque(t, True)
        was_touching = self.touching()
        vy_before = float(self.obs[3])
        self.obs, r, terminated, truncated, _ = self.env.step(int(action))
        self.action = int(action)
        self.drift = self.sensor.update(self.obs, self.action)
        self.reward += float(r)
        self.steps += 1
        if not was_touching and self.touching() and self.touch_speed is None:
            self.touch_speed = abs(vy_before)
        if self.boulder:
            hit = _boulder_hit(u)
            c = _clearance(u, self.boulder)
            self.min_clearance = min(self.min_clearance, c)
            if hit:
                if not terminated:      # a leg on the boulder: the env doesn't know, count it as the crash it is
                    self.reward += -100.0
                self.outcome = 'boulder'
                return True
        if terminated:
            self.outcome = 'landed' if r > 0 else 'crashed'
            return True
        if truncated or self.steps >= MAX_STEPS:
            self.outcome = 'timeout'
            return True
        return False

    def result(self):
        landed = self.outcome == 'landed'
        x = float(self.obs[0])
        on_pad = abs(x) <= PAD_HALF_WIDTH
        return {
            'seed': self.seed, 'reward': self.reward, 'steps': self.steps, 'outcome': self.outcome,
            'landed': landed and on_pad,       # "landed" = came to rest between the flags
            'rested': landed, 'x_final': x,
            'touch_speed': self.touch_speed, 'clearance': None if self.boulder is None else self.min_clearance,
        }


def run_episode(policy, seed, world=World(), env=None, record=None):
    """Fly one episode with policy(inputs, flight) -> action. record: optional callback(flight) each step
    (for videos). Returns Flight.result() plus 'fitness' (env score + landing and centre bonus)."""
    own = env is None
    if own:
        env = make_env()
    f = Flight(env, seed, world)
    if hasattr(policy, 'reset'):
        policy.reset()
    done = False
    while not done:
        if record:
            record(f)
        done = f.step(policy(f.inputs(), f))
    if record:
        record(f)
    out = f.result()
    out['fitness'] = out['reward'] + (LANDING_BONUS + centre_bonus(out['x_final']) if out['landed'] else 0.0)
    if own:
        env.close()
    return out


def centre_bonus(x):
    return CENTRE_BONUS * max(0.0, 1.0 - abs(x) / PAD_HALF_WIDTH)


def _remove_boulder(env):
    u = env.unwrapped
    b = getattr(u, 'boulder_body', None)
    if b is not None:
        u.world.DestroyBody(b)
        u.boulder_body = None


def _add_boulder(env, x, y, r):
    u = env.unwrapped
    body = u.world.CreateStaticBody(position=(x, y))
    body.CreateFixture(shape=circleShape(radius=r), density=0, friction=0.3)
    body.color1, body.color2 = (120, 110, 100), (80, 70, 60)
    u.boulder_body = body


def _boulder_hit(u):
    b = u.boulder_body
    for body in [u.lander] + list(u.legs):
        for ce in body.contacts:
            if ce.other == b and ce.contact.touching:
                return True
    return False


def _clearance(u, boulder):
    """Distance from the lander's centre to the boulder's surface, minus ~the lander's half size."""
    bx, by, r = boulder
    p = u.lander.position
    return math.hypot(p.x - bx, p.y - by) - r - 0.6
