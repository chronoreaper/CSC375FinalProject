import gym
from env import ClawEnv
from absl import app
import matplotlib.pyplot as plt
from stable_baselines3 import DQN

env = ClawEnv(renders=True, isDiscrete=True, removeHeightHack=False, maxSteps=20)
def plot_rewards(running_rewards):
    plt.plot(running_rewards)
    plt.draw()
    plt.pause(0.01)
    pass

def main(argv):

  model = DQN("MlpPolicy", env, verbose=1, buffer_size=10000)
  model.action_space = env.action_space
  model.learn(total_timesteps=1000, log_interval=4)

  obs = env.reset()

  running_rewards = []
  
  while True:
      action, _states = model.predict(obs, deterministic=True)
      obs, reward, done, info = env.step(action)
      env.render()
      if done:
        obs = env.reset()
      
      running_rewards.append(reward)

      plot_rewards(running_rewards)

if __name__ == '__main__':
  app.run(main)
