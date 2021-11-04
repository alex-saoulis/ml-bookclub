from copy import deepcopy
from sarsa import SarsaAgent
from sarsa_fa import SarsaAgent_fa
from easy21 import Easy21, State, Action
import pickle

import numpy as np

import matplotlib.pyplot as plt

NUM_EPISODES = 1000
NUM_EPISODES_2 = 80000

with open("./data/MCActionValue.pkl","rb") as f:
    MCActionValueDict = pickle.load(f)

MCActionValue = np.zeros((10,21))
for state, actions in MCActionValueDict.items():
    if state.playerScore >= 1 and state.playerScore <= 21:
        MCActionValue[state.dealerFirstScore -1 ,state.playerScore -1] = max(actions)

def calculateMeanSquareError(actionValues):
    tsq = np.linalg.norm(actionValues - MCActionValue)**2
    return tsq/actionValues.size

def getActionValueFunction(lambdaIn, numEpisodes):

    env = Easy21()
    sarsaAgent = SarsaAgent(env, lambdaIn)

    historicQs = sarsaAgent.trainAgent(numSteps=numEpisodes)
    actionValues = getActionValuesFromDict(sarsaAgent.actionValueFunction)
    return actionValues, historicQs


def getActionValueFunctionFA(lambdaIn, numEpisodes):

    env = Easy21()
    sarsaAgent = SarsaAgent_fa(env, lambdaIn)

    historicQs = sarsaAgent.trainAgent(numSteps=numEpisodes)
    actionValues = getActionValuesFromWeights(sarsaAgent.weights)
    return actionValues, historicQs

def getActionValuesFromWeights(weights):

    dummyAgent = SarsaAgent_fa("","")

    actionValues = np.zeros((10,21))
    for playerScore in range(1,22):
        for dealerScore in range(1,11):
            featureVectorHit = dummyAgent.generateFeatureVector(State(dealerScore, playerScore), Action.HIT)
            featureVectorStick = dummyAgent.generateFeatureVector(State(dealerScore, playerScore), Action.STICK)
            maxQ = max(np.dot(weights, featureVectorHit), np.dot(weights, featureVectorStick))
            actionValues[dealerScore - 1, playerScore - 1] = maxQ
            
    return actionValues

def getActionValuesFromDict(sarsaAgentDict):
    actionValues = np.zeros((10,21))
    for state, actions in sarsaAgentDict.items():
        if state.playerScore >= 1 and state.playerScore <= 21:
            actionValues[state.dealerFirstScore -1 ,state.playerScore -1] = max(actions)
    return actionValues

lambdas = np.linspace(0,1,21)

msqs = []
stddevs = []
historicQsDct = {}

for lambdaIn in lambdas:
    msqs_inner = []
    for i in range(20):
        actionValues, historicQs = getActionValueFunctionFA(lambdaIn, NUM_EPISODES)
        msqs_inner.append(calculateMeanSquareError(actionValues))
    stddevs.append(np.std(msqs_inner))
    msqs.append(np.mean(msqs_inner))

for lambdaIn in [0,0.5,1]:
    actionValues, historicQs = getActionValueFunctionFA(lambdaIn, NUM_EPISODES_2)
    historicQsDct[lambdaIn] = historicQs
    savedQ = historicQs[-1]
from matplotlib.gridspec import GridSpec


fig = plt.figure(figsize=(10,10))
gridSpec = GridSpec(nrows=2 , ncols=2)

#fig = plt.figure()
ax0 = fig.add_subplot(gridSpec[0,0])
ax0.set_title("MSE after 1000 episodes")
ax0.set_xlabel("Lambda")
ax0.set_ylabel("Mean square error in Q(s,a)")
ax0.plot(lambdas, msqs)
ax0.fill_between(lambdas, list(np.array(msqs) - np.array(stddevs)), list(np.array(msqs) + np.array(stddevs)), alpha=0.2)


ax1 = fig.add_subplot(gridSpec[1,0])
ax1.set_xlabel("Episode number")
ax1.set_ylabel("Mean square error in Q(s,a)")
for lambdaIn, historicQsDict in historicQsDct.items():
    historicQs = list(map(getActionValuesFromWeights, historicQsDict))
    msqs = list(map(calculateMeanSquareError, historicQs))
    ax1.plot(50*np.array(list(range((NUM_EPISODES_2//50) -1))), msqs, label = f"lambda = {lambdaIn}")

    ax1.legend()

ha = fig.add_subplot(gridSpec[:,1], projection='3d')

X = range(1,11, 1)
Y = range(1, 22, 1)

XM, YM = np.meshgrid(X, Y)  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(XM, YM, np.transpose(getActionValuesFromWeights(savedQ)))


plt.show()
