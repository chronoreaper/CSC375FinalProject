import gym
from env import ClawEnv
from absl import app
import matplotlib.pyplot as plt
import tensorflow as tf
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os
import matplotlib.pyplot as plt 

env = ClawEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20)

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)
model.action_space = env.action_space
model.learn(total_timesteps=1000, log_interval=4)

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
plt.savefig('fig.png')
  
  

