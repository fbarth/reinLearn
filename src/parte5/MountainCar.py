import gym
import numpy as np
import matplotlib.pyplot as plt
from QLearningBox import QLearningBox

#
# Reference: https://www.gymlibrary.ml/environments/classic_control/mountain_car/

env = gym.make('MountainCar-v0')
env.reset()

print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

print(env.observation_space.low)
print(env.observation_space.high)

qlearn = QLearningBox(env, 0.1, 0.9, 0.8, 0, 0.999, 1000)
qtable = qlearn.train('data/q-table-mountain-car.csv', 'results/rewards_MountainCar-v0')

state = env.reset()
done = False

while not done:
    env.render()
    state_adj = qlearn.transform_state(state)
    action = np.argmax(qtable[state_adj[0], state_adj[1]])
    state2, reward, done, _ = env.step(action)
    state = state2

input("enter a key...")
env.close()

# Plot Rewards
 
#
#plt.plot(100*(np.arange(len(qtd_actions)) + 1), qtd_actions)
#plt.xlabel('Episodes')
#plt.ylabel('# Actions')
#plt.title('# Actions vs Episodes')
#plt.savefig('results/actions_MountainCar-v0.jpg')     
#plt.close()  