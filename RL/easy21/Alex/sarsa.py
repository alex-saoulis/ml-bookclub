from easy21 import Easy21, State, Card, Colour, Action
import numpy as np  
import random as rand
from copy import deepcopy

class SarsaAgent:

    def __init__(self, env, lambdaIn):
        
        self._lambda = lambdaIn
        self.env = env

        self.stateCounterLookup = {}
        self.stateActionCounterLookup = {}
        self.eligibilityTraces = {}

        self.actionValueFunction = {}

        self.initialiseLookups()

    def initialiseLookups(self):

        for playerScore in range(-9,32):
            for dealerFirstScore in range(1,11):
                state = State(dealerFirstScore, playerScore)

                self.stateCounterLookup[state] = 0
                self.stateActionCounterLookup[state] = [0,0]

                self.actionValueFunction[state] = [0,0]
    
    def initialiseEligibilityTraces(self):
        for playerScore in range(-9,32):
            for dealerFirstScore in range(1,11):
                state = State(dealerFirstScore, playerScore)
                self.eligibilityTraces[state] = [0,0]
    
    def trainAgent(self, numSteps = 100):

        historicQs = []
        for i in range(numSteps):
            
            states, actions, reward = self.runEpisode(self.env)
        
            if (i%50) == 0 and i > 10:
                historicQs.append(deepcopy(self.actionValueFunction))
        return historicQs

    def runEpisode(self, env : Easy21):

            self.initialiseEligibilityTraces()

            actions = []
            states = []

            state = State(env.drawBlackCard().getValue(), env.drawBlackCard().getValue())
            action = self.chooseAction(state)

            while not state.terminal:

                states.append(state)
                actions.append(action)
                self.stateCounterLookup[state] +=1
                self.stateActionCounterLookup[state][action.value] +=1

                oldActionValue = self.actionValueFunction[state][action.value]

                newState = env.step(state, action)

                newAction = self.chooseAction(newState)
                if newState.terminal:
                    newActionValue = 0
                else:
                    newActionValue = self.actionValueFunction[newState][newAction.value]

                delta = newState.reward + newActionValue - oldActionValue
                self.eligibilityTraces[state][action.value] +=1

                for (pastState, pastAction) in zip(states, actions):
                    
                    stepsize = 1/self.stateActionCounterLookup[pastState][pastAction.value]
                    self.actionValueFunction[pastState][pastAction.value] += stepsize * delta * self.eligibilityTraces[pastState][pastAction.value]
                    self.eligibilityTraces[pastState][pastAction.value] *= self._lambda 
                
                state = newState
                action = newAction


            return states, actions, state.reward
    
    def chooseAction(self, state: State):
        actionLookup = {0:Action.STICK, 1: Action.HIT}

        actionValues = np.array(self.actionValueFunction[state])
        maxIndex = np.random.choice(np.where(actionValues == max(actionValues))[0])

        random = rand.random()
        epsilon = 5/(5 + self.stateCounterLookup[state])
        if random > epsilon:
            return actionLookup[maxIndex]
        else:
            return actionLookup[1-maxIndex]


        