import gym
from env import ClawEnv
from absl import app
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt 
from stable_baselines3.common import results_plotter



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = results_plotter.ts2xy(results_plotter.load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.savefig('fig_dqn.png')
    plt.show()
    plt.savefig('fig.png')


log_dir = os.path.join(os.getcwd(), "log")
os.makedirs(log_dir, exist_ok=True)

env = ClawEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)
env = Monitor(env, log_dir)

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)
model.action_space = env.action_space
model.learn(total_timesteps=1000, log_interval=10)

# Helper from the library
results_plotter.plot_results(["./log"], 1000, results_plotter.X_TIMESTEPS, "DQN Rewards")
plot_results(log_dir)

rewards = []


obs = env.reset()
for _ in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    env.render()
    if done:
      obs = env.reset()

plt.plot(rewards)
plt.title('Rewards')
plt.xlabel('Episode')
plt.show()
plt.savefig('fig_dqn1.png')
  
  

