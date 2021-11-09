from mountain_car import MountainCar
from actor_critic import ActorCriticAgent
import numpy as np

env = MountainCar()
acAgent = ActorCriticAgent(env, lambdaIn = 0.99)

acAgent.loadWeights("./weights/debug.pkl")
print(acAgent.w)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,10))
ha = fig.add_subplot(projection='3d')

X = np.linspace(-1.2,0.5,30)
Y = np.linspace(-1.2,1.2,30)

Z = np.zeros((30,30))
meanAction = np.zeros((30,30))


for (i,x) in enumerate(X):
    for (j,y) in enumerate(Y):
        state = (x,y)
        Z[i,j] = acAgent.getValueFunction(state)
        meanAction[i,j] = np.dot(acAgent.generateFeatureVector(state), acAgent.theta)
XM, YM = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(XM, YM, np.transpose(meanAction))
print(acAgent.theta)
plt.show()