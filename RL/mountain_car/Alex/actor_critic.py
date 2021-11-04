import numpy as np
class ActorCriticAgent():

    def __init__(self, action_spread = 0.1):
        self.theta = np.zeros((64))
        self.w = np.zeros((64))
        self.action_sigma = action_spread

    def getAction(self, state):
        # evaluate the policy (actor):
        # we will use a Gaussian policy ->
        # continuous action a ~ N(theta*x, sigma**2)
        featureVector = self.generateFeatureVector(state)
        return np.random.normal(np.dot(featureVector, self.theta), self.action_sigma**2)


    def generateFeatureVector(self, state):
        position, velocity = state
        rbf_xpositions = [-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4]
        rbf_velocities = np.linspace(-1.2, 1.2, 8)

        sig_pos = 0.1

        featureVector = np.zeros((64))
        for i,xpos in enumerate(rbf_xpositions):
            for j,vel in enumerate(rbf_velocities):
                featureVector[i*8+j] = np.exp(-((position-xpos)/(2*sig_pos))**2)*np.exp(-((velocity-vel)/(2*sig_pos))**2)

        return featureVector

acAgent = ActorCriticAgent()
state = [-0.9, -1]
print(acAgent.generateFeatureVector(state))