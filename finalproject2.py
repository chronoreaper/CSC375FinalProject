from email import policy
import itertools
import gym
import math
import random
import numpy as np
from absl import app
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, namedtuple
import collections
from env import ClawEnv
from gym import spaces
import pybullet as p
import tensorflow as tf

env = ClawEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20, dv=0.06)

from typing import Any, List, Sequence, Tuple
from models import ActorCriticPolicy
from agents import ActorCriticAgent

# Create the environment
min_episodes_criterion = 100
max_episodes = 10000
steps = 500
gamma = 0.9

# Set seed for experiment reproducibility
seed = 42
env.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


def normalize(state):
    state = state / 255.
    return state

def main(argv):
    discount_factor = 1.0
    alpha = 0.5

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    policy = GreedyPolicy(Q, eps. env.action_space.n)

    for i in tf.range(max_episodes):
        state = env.reset()

        for j in tf.range(steps):

            state = normalize(state)
            action_prob = policy(state)

            # Choose an action based on probability
            action = np.random.choice(np.arange(len(action_prob)), p = action_prob)

            # Run the program
            next_state, reward, done, _ = env.step(action)

            # Update the Q state
            best_next_action = np.argmax(Q[next_state])
            target_reward = reward + discount_factor * Q[next_state][best_next_action]
            delta_reward = target_reward - Q[next_state][best_next_action]
            Q[state][action] += alpha * delta_reward

            if done:
                break

            state = next_state

        if i % 10 == 0:
            tf.print('total reward after {} episodes is {}'.format(i, running_reward))

# Reference https://www.geeksforgeeks.org/q-learning-in-python/
def GreedyPolicy(Q, num_actions):
    """
    """
    def policy(state):
        probability = np.ones(num_actions, dtype=float) * eps / num_actions
        best_action = np.argmax(Q[state])
        probability[best_action] += (1.0 - eps)
        return probability
    return policy

if __name__ == '__main__':
  app.run(main)
