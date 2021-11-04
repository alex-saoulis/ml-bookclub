from easy21 import Easy21, State, Card, Colour, Action
import numpy as np  
import random as rand
from copy import deepcopy

class SarsaAgent_fa:

    def __init__(self, env, lambdaIn):
        
        self._lambda = lambdaIn
        self.env = env

        self.epsilon = 0.05
        self.alpha = 0.01

        self.coursePlayerRanges = [[1,6],[4,9],[7,12],[10,15],[13,18],[16,21]]
        self.courseDealerRanges = [[1,4],[4,7],[7,10]]

        self.eligibilityTraces = np.zeros((36))

        self.weights = np.zeros((36))
    
    def generateFeatureVector(self, state : State, action : Action):
        featureVector = np.zeros((36))
        playerRangeLength = len(self.coursePlayerRanges)

        if action is Action.HIT:
            actionShift = 0
        else:
            actionShift = 18

        for i, dealerRange in enumerate(self.courseDealerRanges):
            if state.dealerFirstScore >= dealerRange[0] and state.dealerFirstScore <=dealerRange[1]:
                for j, playerRange in enumerate(self.coursePlayerRanges):
                    if state.playerScore >= playerRange[0] and state.playerScore <= playerRange[1]:
                        featureVector[i*playerRangeLength + j + actionShift] =1
        
        return featureVector
    
    def initialiseEligibilityTraces(self):
        self.eligibilityTraces = np.zeros((36))
    
    def trainAgent(self, numSteps = 100):

        historicWeights = []
        for i in range(numSteps):
            
            reward = self.runEpisode(self.env)
        
            if (i%50) == 0 and i > 10:
                historicWeights.append(deepcopy(self.weights))
        return historicWeights

    def runEpisode(self, env : Easy21):

            self.initialiseEligibilityTraces()

            state = State(env.drawBlackCard().getValue(), env.drawBlackCard().getValue())
            action = self.chooseAction(state)
            featureVector = self.generateFeatureVector(state, action)

            while not state.terminal:

                oldActionValue = np.dot(self.weights, featureVector)

                newState = env.step(state, action)
                newAction = self.chooseAction(newState)

                newfeatureVector = self.generateFeatureVector(newState, newAction)

                if newState.terminal:
                    newActionValue = 0
                else:
                    newActionValue = np.dot(self.weights, newfeatureVector)

                delta = newState.reward + newActionValue - oldActionValue

                self.eligibilityTraces *= self._lambda
                self.eligibilityTraces += featureVector
                    
                self.weights += self.alpha * delta * self.eligibilityTraces
                
                state = newState
                action = newAction

                featureVector = newfeatureVector 
            return state.reward
    
    def chooseAction(self, state: State):

        actionLookup = {0:Action.STICK, 1: Action.HIT}

        actionValues = np.array([np.dot(self.weights, self.generateFeatureVector(state, Action.STICK)),
                                  np.dot(self.weights, self.generateFeatureVector(state, Action.HIT))  ])

        maxIndex = np.random.choice(np.where(actionValues == max(actionValues))[0])

        random = rand.random()
        if random > self.epsilon:
            return actionLookup[maxIndex]
        else:
            return actionLookup[1-maxIndex]


        