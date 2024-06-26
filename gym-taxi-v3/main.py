from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make("Taxi-v3")
agent = Agent(alpha=1)
avg_rewards, best_avg_reward = interact(env, agent)
