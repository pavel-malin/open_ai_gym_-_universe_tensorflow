# the robot learns to walk
import gym

env = gym.make('BipedalWalker-v2')

for i_episode in range(100):
    observation = env.reset()
for t in range(1800):
    env.render()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        print("{} timesteps for the episode".format(t+3))
        break

