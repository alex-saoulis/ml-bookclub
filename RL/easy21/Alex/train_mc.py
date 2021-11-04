from MCcontrol import MCAgent
from easy21 import Easy21, State
import pickle

import numpy as np

env = Easy21()
mcAgent = MCAgent(env)

means = mcAgent.trainAgent(numSteps=10000000)

with open("./data/MCActionValue.pkl", "wb") as fout:
    pickle.dump(mcAgent.actionValueFunction, fout)

import matplotlib.pyplot as plt

fig = plt.figure()
plt.subplot(2,1,1)
plt.xlabel("Number of episodes")
plt.ylabel("Mean reward over 100000 episodes")
plt.plot(100000*np.array(list(range(len(means)))), means)


ha = fig.add_subplot(212, projection='3d')

actionValues = np.zeros((10,21))
for state, actions in mcAgent.actionValueFunction.items():
    actionValues[state.dealerFirstScore - 1,state.playerScore - 1] = max(actions)

X = range(1,11, 1)
Y = range(1, 22, 1)

XM, YM = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(XM, YM, np.transpose(actionValues))

plt.show()