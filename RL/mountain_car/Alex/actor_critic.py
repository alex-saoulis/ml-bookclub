import pickle
import numpy as np
from tqdm import tqdm

NUM_GRIDPOINTS = 6
VEC_SIZE = NUM_GRIDPOINTS**2

class ActorCriticAgent():

    def __init__(self, env, action_spread = 0.5, lambdaIn = 0.9):

        self.env = env
        self.count = 0
        self.theta = np.random.normal(0,0.1,(VEC_SIZE))
        self.w =np.random.normal(0,0.1,(VEC_SIZE))
        self.action_sigma = action_spread

        self.visitCounterVector = np.zeros((VEC_SIZE))

        self._lambda = lambdaIn

        self.initEligibilityTraces()
        self.initRBFReferenceVectors()
    
    def initRBFReferenceVectors(self):

        rbf_xpositions = np.linspace(-1.2, 0.5, NUM_GRIDPOINTS)
        rbf_velocities = np.linspace(-1.2, 1.2, NUM_GRIDPOINTS)

        self.xposRBFVector = np.concatenate([[rbf_xpos for i in range(NUM_GRIDPOINTS)] for rbf_xpos in rbf_xpositions])
        self.velRBFVector = np.concatenate([[rbf_vel for rbf_vel in rbf_velocities] for i in range(NUM_GRIDPOINTS)])

    def initEligibilityTraces(self):

        self.actorEligibilityTrace = np.zeros((VEC_SIZE))
        self.criticEligibilityTrace = np.zeros((VEC_SIZE))

    def getAction(self):
        # evaluate the policy (actor):
        # we will use a Gaussian policy ->
        # continuous action a ~ N(theta*x, sigma**2)
        sampledAction = np.random.normal(np.dot(self.featureVector, self.theta), self.action_sigma**2)
        return  min(max(sampledAction,-0.2),0.2)
    
    def getScoreFunction(self, action):
        return (action - min(max(np.dot(self.featureVector, self.theta),-0.2),0.2))*self.featureVector/(self.action_sigma**2)

    def getValueFunction(self, state):
        featureVector = self.generateFeatureVector(state)
        return np.dot(self.w, featureVector)

    def generateFeatureVector(self, state):
        position, velocity = state

        sig_pos = 2*0.2**2
        xposVector = np.ones((VEC_SIZE))*position
        velVector = np.ones((VEC_SIZE))*velocity

        # featureVector = np.zeros((64))
        # for i,xpos in enumerate(rbf_xpositions):
        #     for j,vel in enumerate(rbf_velocities):
        #         featureVector[i*8+j] = np.exp(-((position-xpos)/(2*sig_pos))**2)*np.exp(-((velocity-vel)/(2*sig_pos))**2)

        xposFeatureVector = np.exp(-((np.subtract(xposVector,self.xposRBFVector)**2/(2*sig_pos))))
        velFeatureVector = np.exp(-((np.subtract(velVector,self.velRBFVector)**2/(2*sig_pos))))
        featureVector = np.multiply(xposFeatureVector, velFeatureVector)
        norm = np.linalg.norm(featureVector)
        return featureVector/norm

    def train(self, numSteps = 100):
        self.historicalReward = []
        for i in tqdm(range(numSteps)):
            if i == 10000:
                self.action_sigma = 0.1
            r = self.runEpisode()
            self.historicalReward.append(r)
        
    def runEpisode(self, exploringStarts = True):

        totReward = 0
        self.initEligibilityTraces()
        if exploringStarts:
            state = self.env.reset()
        else:
            state = self.env.reset(initial_position = 0.5)
        done = False

        while not done:

            self.featureVector = self.generateFeatureVector(state)
            action = self.getAction()
            oldValue = self.getValueFunction(state)

            newState, r, done = self.env.step(action)
            totReward += r

            if done:
                newValue = 0
            else:
                newValue = self.getValueFunction(newState)      
            
            delta = r + newValue - oldValue
            alpha = self.getAlpha()

            self.updateEligibilityTraces(action)

            self.updateCritic(delta, alpha)
            self.updateActor(delta, alpha)

            state = newState
            
        return totReward


    def updateEligibilityTraces(self,action):
        self.criticEligibilityTrace *= self._lambda
        self.criticEligibilityTrace += self.featureVector

        self.actorEligibilityTrace *= self._lambda
        self.actorEligibilityTrace += self.getScoreFunction(action)
        # if self.count %1000 == 0:
        #     print(self.getScoreFunction(state,action))
        #     print(self.theta)

    def updateCritic(self, delta, alpha):
        self.w +=  alpha * delta * self.criticEligibilityTrace
    
    def updateActor(self, delta, alpha):
        self.theta += alpha * delta * self.actorEligibilityTrace
        self.count +=1

            #

    def getAlpha(self):
        #return 1/(100+np.dot(self.visitCounterVector, self.featureVector))
        return 0.001
    
    def saveWeights(self, filepath):
        dct = {"w": self.w, "theta": self.theta}
        with open(filepath, 'wb') as file:
            pickle.dump(dct, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def loadWeights(self, filepath):
        with open(filepath, 'rb') as handle:
            weightsDct = pickle.load(handle)
        
        self.w = weightsDct["w"]
        self.theta = weightsDct["theta"]