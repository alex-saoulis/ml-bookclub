import random as rand
from enum import Enum
from copy import copy, deepcopy

class Action(Enum):
    STICK = 0
    HIT = 1

class Colour(Enum):
    BLACK = 1
    RED = 2

class Card:
    def __init__(self, colour : Colour, value):
        self.colour = colour
        self.value = value

    def getValue(self):
        if self.colour is Colour.BLACK:
            return self.value
        else:
            return -1*self.value

class State:

    def __init__(self, dealerFirstScore, playerScore):

        self.dealerFirstScore = dealerFirstScore
        self.playerScore = playerScore
        self.reward = 0

        self.checkIfStateTerminal()

    def checkIfStateTerminal(self):
        if self.playerScore < 1 or self.playerScore > 21:
            self.reward = -1
            self.terminal = True
        else:
            self.terminal = False

    def __key(self):
        return (self.dealerFirstScore, self.playerScore)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, State):
            return self.__key() == other.__key()

class Easy21:

    def __init__(self, verbose = False):
        self.verbose = verbose

    def step(self, state : State, action):

        if self.verbose:
            self.render(state, action)

        if action is Action.HIT:
            nextCard = self.drawCard()
            newState = State(state.dealerFirstScore, state.playerScore + nextCard.getValue())
            return newState
        elif action is Action.STICK:
            state.terminal = True
            state.reward = self.dealerPlaysAndReturnsReward(deepcopy(state))
            return state
        
    
    def dealerPlaysAndReturnsReward(self, state: State):
        dealerScore = state.dealerFirstScore
        while True:
            nextCard = self.drawCard()
            dealerScore += nextCard.getValue()
            if dealerScore < 1 or dealerScore > 21:
                if self.verbose:
                    print(f"Dealer went bust at {dealerScore}!")
                return 1
            elif dealerScore >= 17:
                if self.verbose:
                    print(f"Dealer stuck at {dealerScore}.")
                if dealerScore < state.playerScore:
                    return 1
                elif dealerScore > state.playerScore:
                    return -1
                else:
                    return 0


    def drawBlackCard(self):
        return Card(Colour.BLACK, rand.randrange(1,11))

    def drawCard(self):
        return Card(self.getRandomColour(), rand.randrange(1,11))

    def getRandomColour(self):
        random = rand.random()
        if random <= 1/3:
            return Colour.RED
        else:
            return Colour.BLACK

    def render(self, state : State, action):
        actions = {0:"stick", 1:"hit"}
        
        print(f"State:\n\tDealer first card: {state.dealerFirstScore}\n\tPlayer score: {state.playerScore}")
        if not state.terminal:
            print(f"Player chose to {actions[int(action.value)]}.")