import gym
import math
import random
import numpy as np
from absl import app
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
from env import ClawEnv
from gym import spaces
import pybullet as p
import tensorflow as tf

env = ClawEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)

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

def plot_rewards(running_rewards):
    plt.plot(running_rewards)
    plt.draw()
    plt.pause(0.01)
    pass

def normalize(state):
    state = state / 255.
    return state

def main(argv):

    model = ActorCriticPolicy(input_shape=env.observation_space.shape, action_dim=env.action_space.n)
    agent = ActorCriticAgent(model=model)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)

    episodes_reward = []
    running_rewards = []

    for i in tf.range(max_episodes):
        done = False
        state = env.reset()
        episode_reward = 0
        log_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        with tf.GradientTape() as tape:
            for j in tf.range(steps):
                state = normalize(state)
                action, action_log_probs, value = agent.act(state)
                log_probs = log_probs.write(j, action_log_probs[0, action])
                values = values.write(j, tf.squeeze(value))

                next_state, reward, done, _ = env.step(action)

                rewards = rewards.write(j, reward)

                if done:
                    break

                state = next_state
            log_probs = log_probs.stack()
            values = values.stack()
            rewards = rewards.stack()

            returns = agent.compute_expected_return(rewards, gamma)
            loss = agent.compute_loss(log_probs, returns, values)

        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        episode_reward = tf.math.reduce_sum(rewards)

        episodes_reward.append(float(episode_reward))
        running_reward = tf.math.reduce_mean(episodes_reward)
        running_rewards.append(running_reward)

        if i % 10 == 0:
            tf.print('total reward after {} episodes is {}'.format(i, running_reward))

        plot_rewards(running_rewards)

if __name__ == '__main__':
  app.run(main)
