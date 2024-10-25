import numpy as np
import gym
import random

class EthicalDilemmaEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(2)
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == 0:
            if action == 0:
                reward = 10
                self.state = 1
            elif action == 1:
                reward = -10
                self.state = 1
        else:
            reward = 0

        done = True
        return self.state, reward, done, {}

    def render(self):
        if self.state == 0:
            print("In initial state, agent must choose an action.")
        else:
            print("The agent has made a choice and faces the consequences.")

env = EthicalDilemmaEnv()
q_table = np.zeros([env.observation_space.n, env.action_space.n])

alpha = 0.1
gamma = 0.9
epsilon = 0.1
num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        state = next_state

print("Trained Q-table:")
print(q_table)
