
# stolen from Kathryn
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from environment_constants import *


def getFirstCard():
    return Card(COLOURS.BLACK)


def getCard():
    return Card(getColour())


def getColour():
    decision_value = np.random.uniform(0, 1.0)
    return COLOURS.BLACK if decision_value > 1/3 else COLOURS.RED


class Card():
    def __init__(self, colour):
        self.value = np.random.randint(
            CARD_VALUES.MIN, CARD_VALUES.MAX + CARD_VALUES.MIN)
        self.colour = colour


class State():
    def __init__(self, dealer_card: Card, player_value: int):
        self.dealer_card = dealer_card
        self.player_value = player_value


def getCardValue(card: Card) -> int:
    return card.value if card.colour == COLOURS.BLACK else (-1 * card.value)


def ifScoreBust(score):
    return score >= BRACKETS.BUST_UP or score <= BRACKETS.BUST_DOWN


def giveRewards(score1: int, score2: int):
    if score1 == score2:
        return REWARDS.NEUTRAL
    elif score1 > score2:
        return REWARDS.NEGATIVE
    return REWARDS.POSITIVE


class Easy21Env():
    def __init__(self):
        self.reset()

    def step(self, hit: bool):
        if hit:
            new_card: Card = getCard()
            self.player_score += getCardValue(new_card)
            if ifScoreBust(self.player_score):
                self.player_reward = REWARDS.NEGATIVE
                self.done = True
                self.state = State(self.dealer_card.value, self.player_score)
            else:
                self.state = State(self.dealer_card.value, self.player_score)
        else:
            self.dealer_round()
            if ifScoreBust(self.dealer_score):
                self.player_reward = REWARDS.POSITIVE
            else:
                self.player_reward = giveRewards(
                    self.dealer_score, self.player_score)
            self.done = True
        return self.state

    def dealer_round(self):
        self.dealer_score = self.dealer_card.value
        while self.dealer_score < BRACKETS.DEALER_STICK and self.dealer_score > BRACKETS.BUST_DOWN:
            new_card = getCard()
            self.dealer_score += getCardValue(new_card)

    def reset(self):
        player_card = getFirstCard()
        dealer_card = getFirstCard()

        self.player_score = player_card.value
        self.dealer_card = dealer_card

        self.state = State(self.dealer_card.value, self.player_score)
        self.done = False
        self.player_reward: REWARDS = None
        self.dealer_score: int = self.dealer_card.value
