from mountain_car import MountainCar
import numpy as np

env = MountainCar()
env.reset()

def generateRandomAction():
    return np.random.uniform(-0.2,0.2)

done = False
while not done:
    o, r, done = env.step(generateRandomAction())

env.render("./dummy.gif","gif")
