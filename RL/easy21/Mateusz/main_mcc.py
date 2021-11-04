from src.agent import Agent_MC
from src.easy21 import easy21_game

from tqdm import tqdm
import numpy as np



env = easy21_game(aces = False)
agent = Agent_MC(n0 = 100)



reward = 0
rew_arr = []
for q in tqdm(range(100000)):


    sa_pairs = []
    env.reset()

    while not env.term:
        
        s = env.get_state()                         # get current state
        a = agent.action(s)                         # get action
        
        o,r,d,i = env.step(a)                       # step

        agent.updateN(s,a)                          # update n

        sa_pairs.append((s,a))                      # append to data

    reward += r
    
    if q%10000 == 0 and q != 0:
        rew_arr.append(reward/10000)
        reward = 0
    

    for state, action in sa_pairs:                  # once episode is over process the pairs

        agent.updateQ(state,action,r)               # update Q

agent.updateV()


# copied from https://stackoverflow.com/questions/11409690/plotting-a-2d-array-with-mplot3d 


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set up grid and test data

nx, ny = 10,21

x = range(nx)
y = range(ny)

data = agent.V.copy()

print(data)
print(data.shape)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')

X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D

ha.plot_surface(X, Y, data)

plt.show()



plt.plot(rew_arr)
plt.show()
    
    
import pickle as pkl

with open('mcc_Q.pkl','wb') as f:
    pkl.dump(data, f)
    



