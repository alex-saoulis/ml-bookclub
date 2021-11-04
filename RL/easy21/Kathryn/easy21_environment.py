import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Action(Enum):
    HIT = 0
    STICK = 1


class FirstCard():
    def __init__(self):
        self.colour = 'black'
        self.value = np.random.randint(1, 11)


class Card(FirstCard):
    def __init__(self):
        super().__init__()
        self.colour = self.assign_colour()

    def assign_colour(self):
        decision_value = np.random.uniform(0, 1.0)
        if decision_value > 1/3:
            colour = 'black'
        else:
            colour = 'red'
        return colour


class State():
    def __init__(self, dealer_card, player_value):
        self.dealer_card = dealer_card
        self.player_value = player_value


class Easy21Env():
    def __init__(self):
        self.reset()

    def step(self, action):
        if action == Action.STICK:
            self.dealer_round()
            if self.dealer_total > 21 or self.dealer_total < 1:
                # print('dealer is bust!')
                self.player_reward = 1
            else:
                if self.dealer_total == self.player_score:
                    self.player_reward = 0
                elif self.dealer_total > self.player_score:
                    self.player_reward = -1
                else:
                    self.player_reward = 1
            self.done = True

        elif action is Action.HIT:
            new_card = Card()
            if new_card.colour == 'red':
                self.player_score -= new_card.value
            elif new_card.colour == 'black':
                self.player_score += new_card.value

            if self.player_score > 21 or self.player_score < 1:
                # print('Youve gone bust!')
                self.player_reward = -1
                self.state = State(self.dealer_card.value, self.player_score)
                self.done = True
            else:
                self.state = State(self.dealer_card.value, self.player_score)

        else:
            raise IndexError(
                f'Action {action} is out of the action state range - please choose either stick or hit')
        return self.state

    def dealer_round(self):
        self.dealer_total = self.dealer_card.value
        while self.dealer_total < 17 and self.dealer_total >= 1:
            # fix this - currently the dealer CAN be going below 1!!! move if statements from the logic
            # above into this bit
            new_card = Card()
            if new_card.colour == 'red':
                self.dealer_total -= new_card.value
            elif new_card.colour == 'black':
                self.dealer_total += new_card.value

    def reset(self):
        player_card = FirstCard()
        dealer_card = FirstCard()

        self.player_score = player_card.value
        self.dealer_card = dealer_card

        self.state = State(self.dealer_card.value, self.player_score)
        self.done = False
        self.player_reward = None
        self.dealer_total = self.dealer_card.value
