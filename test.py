import gym
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

model = DDPG.load('biped_Walker.zip')

env = gym.make('BipedalWalker-v3',render_mode='human')
env = DummyVecEnv([lambda: env])

obs = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
