import gym
import random

env = gym.make('Taxi-v3')

alpha = 0.85
gamma = 0.90
epsilon = 0.8

Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0


def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return max(list(range(env.action_space.n)), key=lambda x: Q[(state, x)])


for i in range(4000):
    r = 0
    state = env.reset()
    action = epsilon_greedy(state, epsilon)
    while True:
        nextstate, reward, done, _ = env.step(action)
        nextaction = epsilon_greedy(nextstate, epsilon)
        Q[(state, action)] = alpha * (reward + gamma * Q[(nextstate, nextaction)] - Q[state, action])
        action = nextaction
        state = nextstate
        if done:
            break

env.close()
