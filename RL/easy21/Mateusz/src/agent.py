from .easy21 import easy21_game
import numpy as np
import random


class Agent_MC:

    def __init__(self, n0 = 100):

        self.n0 = n0

        self.N = np.zeros((21,10,2))                # Vist  matrix
        self.Q = np.zeros((21,10,2))                # Policy matrix
        self.V = np.zeros((21,10))                  # Value function

        self.actions = ["stick","hit"]
        self.action_enum = {"hit":1,"stick":0}

        

    def action(self,state):

        p_idx = state[0]-1                      # player
        d_idx = state[1]-1                      # dealer

        eps = self.n0/(self.n0+np.sum(self.N[p_idx,d_idx,:]))

        rand = np.random.random()

        if rand > 1-eps:

            action = random.choice(self.actions) 
        
        else:
            action = self.actions[np.argmax(self.Q[p_idx,d_idx,:])]

        return action 

    def train(self):
        pass

    def updateN(self,state,action):

        self.N[state[0]-1,state[1]-1,self.action_enum[action]]+=1

    def updateQ(self,state,action,reward):


        step = 1.0 / self.N[state[0]-1, state[1]-1, self.action_enum[action]]
        error = reward - self.Q[state[0]-1, state[1]-1, self.action_enum[action]]
        self.Q[state[0]-1, state[1]-1, self.action_enum[action]] += step * error

    def updateV(self):
        self.V = np.amax(self.Q, axis = -1)

    def getBestActionMat(self):
        return np.argmax(self.Q, axis = -1)






class Agent_Sarsa_Lambda(Agent_MC):
 
    def __init__(self, n0 = 100, padding = 20):

        self.n0 = n0
        self.padding = padding
        #                                           These need renaming
        self.N = np.zeros((21+padding,10+padding,2))                # Vist  matrix
        self.Q = np.zeros((21+padding,10+padding,2))                # Policy matrix

        self.E = np.zeros((21+padding,10+padding,2))                # Eligibility matrix

        self.V = np.zeros((21+padding,10+padding))                  # Value Matrix

        self.actions = ["stick","hit"]
        self.action_enum = {"hit":1,"stick":0}


        self.state_tracker = []
        self.action_tracker = []

 
    def get_action_value(self,state, action, d = False):
        
        if d:
            return 0
        else:
            p_idx = state[0]-1 + self.padding//2         # player
            d_idx = state[1]-1 + self.padding//2         # dealer
            return self.Q[p_idx,d_idx,self.action_enum[action]]
 
    def action(self,state):


        

        p_idx = state[0]-1 + self.padding//2                       # player
        d_idx = state[1]-1 + self.padding//2                       # dealer

        # print(state, p_idx,d_idx)

        eps = self.n0/(self.n0+np.sum(self.N[p_idx,d_idx,:]))

        rand = np.random.random()

        if rand > 1-eps:

            action = random.choice(self.actions) 
        
        else:

            action = self.actions[np.argmax(self.Q[p_idx,d_idx,:])]

        return action

    def updateN(self,state,action):

        self.N[state[0]-1 + self.padding//2 ,state[1]-1 + self.padding//2 ,self.action_enum[action]]+=1
 
    def updateE(self,state,action):

        self.E[state[0]-1 + self.padding//2 ,state[1]-1 + self.padding//2 ,self.action_enum[action]]+=1
 
    def updateQ(self,state,action,reward):

        p_idx = state[0]-1 + self.padding//2         # player
        d_idx = state[1]-1 + self.padding//2         # dealer

        step = 1.0 / self.N[p_idx, d_idx, self.action_enum[action]]
        error = reward - self.Q[p_idx, d_idx, self.action_enum[action]]
        self.Q[p_idx, d_idx, self.action_enum[action]] += step * error

    def calc_alpha(self,state,action):

        p_idx = state[0]-1 + self.padding//2         # player
        d_idx = state[1]-1 + self.padding//2         # dealer
        
        Nsa = self.N[p_idx, d_idx, self.action_enum[action]]

        return 1/Nsa

    


        
    