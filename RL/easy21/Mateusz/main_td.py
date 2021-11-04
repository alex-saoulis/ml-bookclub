from src.agent import Agent_MC, Agent_Sarsa_Lambda
from src.easy21 import easy21_game

from tqdm import tqdm
import numpy as np
import pickle as pkl







for lambda_val in [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
    
    print(lambda_val)

    env = easy21_game(aces = False)
    agent = Agent_Sarsa_Lambda(n0 = 10,padding = 32)
    gamma = 0.99

    reward = 0
    rew_arr = []

    mse_arr = []


    with open('mcc_Q.pkl','rb') as f:
        mcc_Q = pkl.load(f)  


    for q in tqdm(range(100000)):


        sa_pairs = []
        env.reset()

        ## init step for sarsa lambda

        s = env.get_state()                         # get current state
        a = agent.action(s)                         # get action
        
        
        agent.updateN(s,a)

        while not env.term:

            sa_pairs.append((s,a))                      # append to data

            o,r,d,i = env.step(a)                       # step


            s_p = env.get_state()                       # get current state
            a_p = agent.action(s_p)                     # get action

            agent.updateE(s,a)                          # update E

            # for s,a in sa_pairs:

            alpha = agent.calc_alpha(s,a)    

            delta = r +  gamma*agent.get_action_value(s_p,a_p, env.term) - agent.get_action_value(s,a)
            
            sa_pairs.append((s,a))                      # append to data

            # for s_,a_ in sa_pairs:

            s_ = s
            a_ = a

            p_idx = s_[0]-1 + 32//2                       # player
            d_idx = s_[1]-1 + 32//2                       # dealer

            agent.Q[p_idx, d_idx, agent.action_enum[a_]] = agent.Q[p_idx, d_idx, agent.action_enum[a_]] + alpha*delta*agent.E[p_idx, d_idx, agent.action_enum[a_]]
            agent.E[p_idx, d_idx, agent.action_enum[a_]] = gamma*lambda_val*agent.E[p_idx, d_idx, agent.action_enum[a_]]

            s = s_p
            a = a_p 
            
            agent.updateN(s,a)
            





        reward += r
        
        if q%1000 == 0 and q != 0:
            agent.updateV()
            mse_arr.append(np.mean((agent.V[32//2:21+32//2,32//2:10+32//2]-mcc_Q)**2))
            rew_arr.append(reward/1000)
            reward = 0

    agent.updateV()


    # copied from https://stackoverflow.com/questions/11409690/plotting-a-2d-array-with-mplot3d 


    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D

    # # Set up grid and test data

    # unpad = 0

    # nx, ny = 10+unpad*2,21+unpad*2

    # x = range(1,nx+1,1)
    # y = range(1,ny+1,1)

    # data = agent.V[32//2-unpad:21+32//2+unpad,32//2-unpad:10+32//2+unpad].copy()

    # print(data)
    # print(data.shape)

    # hf = plt.figure()
    # ha = hf.add_subplot(111, projection='3d')

    # X, Y = np.meshgrid(x, y)  # `plot_surface` expects `x` and `y` data to be 2D

    # ha.plot_surface(X, Y, data)

    # plt.show()



    # plt.plot(rew_arr)
    # plt.show()

    # plt.plot(mse_arr)
    # plt.show()

    with open('mse_sl_V'+str(lambda_val)+'.pkl','wb') as f:
        pkl.dump(mse_arr, f)

    
    

    



