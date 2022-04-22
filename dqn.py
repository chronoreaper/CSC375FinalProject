import gym
from env import ClawEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

env = ClawEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20)

model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)
model.action_space = env.action_space
model.learn(total_timesteps=1000, log_interval=4)

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()