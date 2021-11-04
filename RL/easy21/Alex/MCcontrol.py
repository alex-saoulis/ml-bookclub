from easy21 import Easy21, State, Card, Colour, Action
import numpy as np  
import random as rand


class MCAgent:

    def __init__(self, env):

        self.env = env

        self.stateCounterLookup = {}
        self.stateActionCounterLookup = {}

        self.actionValueFunction = {}

        self.initialiseLookups()

    def initialiseLookups(self):

        for playerScore in range(1,22):
            for dealerFirstScore in range(1,11):
                state = State(dealerFirstScore, playerScore)

                self.stateCounterLookup[state] = 0
                self.stateActionCounterLookup[state] = [0,0]

                self.actionValueFunction[state] = [0,0]
    
    def trainAgent(self, numSteps = 100):
        rewards = []
        means = []
        for i in range(numSteps):
            
            states, actions, reward = self.runEpisode(self.env)
            rewards.append(reward)
            for state, action in zip(states, actions):
                self.stateCounterLookup[state] +=1

                self.stateActionCounterLookup[state][action.value] +=1

                prevAVFunction = self.actionValueFunction[state][action.value]
                stepsize = 1/self.stateActionCounterLookup[state][action.value]

                self.actionValueFunction[state][action.value] += stepsize* (reward - prevAVFunction)
        
            if (i%100000) == 0 and i > 10:
                means.append(np.mean(rewards))
                print(f"rolling average reward over 10000 episodes: {means[-1]}")
                rewards = []
        return means

    def runEpisode(self, env : Easy21):

            reward = None
            actions = []
            state = State(env.drawBlackCard().getValue(), env.drawBlackCard().getValue())
            states = [state]
            while True:
                action = self.chooseAction(state)
                actions.append(action)
                state = env.step(state, action)
                if state.terminal:
                    reward = state.reward
                    break
                else:
                    states.append(state)

            return states, actions, reward
    
    def chooseAction(self, state: State):
        actionLookup = {0:Action.STICK, 1: Action.HIT}

        actionValues = np.array(self.actionValueFunction[state])
        maxIndex = np.random.choice(np.where(actionValues == max(actionValues))[0])

        random = rand.random()
        epsilon = 1000/(1000 + self.stateCounterLookup[state])
        if random > epsilon:
            return actionLookup[maxIndex]
        else:
            return actionLookup[1-maxIndex]


        