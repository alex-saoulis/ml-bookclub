
# stolen from Kathryn
from typing import Dict
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from environment import Easy21Env, State
import random
N0: int = 1000
X0: int = 10
Y0: int = 21


def plot_value_function(agent_history, n_iterations: int):
    data = np.maximum(
        agent_history.value_function[0], agent_history.value_function[1])
    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    x, y = np.meshgrid(range(X0), range(Y0))
    ha.plot_surface(x, y, data)
    ha.set_title(f'value function with {n_iterations} iterations')
    ha.set_xlabel('dealer first card')
    ha.set_ylabel('player score')
    ha.set_zlabel('value')
    plt.savefig("trial_run")
    plt.show()


class Agent():
    def __init__(self):
        self.environment = Easy21Env()
        # the number of times a state is visited (2d array)
        self.state_visits = np.zeros((1, Y0, X0))
        # the number of times an action is used in a state (3d array)
        self.action_state_visits = np.zeros((2, Y0, X0))
        # the policy is implicit in this
        self.value_function = np.zeros((2, Y0, X0))
        self.episode_states = []
        self.episode_actions = []
        # this should be a 3d array where the x axis is the player score, the y axis is the dealer score and the
        # z axis is the action (hit or stick). then, for a given player and dealer score and action, (which you find
        # by navigating through using the indices), you go with the value / action that gives the highest reward
        # the counter matrix should look the same as the other one, but rather than updating with the reward each time,
        # we increment the state-action pairs with the number of times we saw that state in the episode by 1

    def run_optimisation(self, iterations):
        for i in tqdm.tqdm(range(iterations)):
            self.run_episode(i)

    def update_persistent_array(tmp, persistent):
        # https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html
        np.add.at(persistent, tmp, 1)
        pass  # using bincount? need to create them using np
        # and change back action to 0/1 I guess, as then easier numpy integration

    def run_episode(self, episodeNumber):
        self.environment.reset()
        new_state = self.environment.state
        while not self.environment.done:
            self.episode_states.append(new_state)
            action = self.choose_action(self.environment.state)
            new_state = self.environment.step(action)
            self.episode_actions.append(action)
            # something like self.episode_history containing both

        for state in self.episode_states:
            self.state_visits[0, state.player_value -
                              1, state.dealer_card-1] += 1

        for state, action in zip(self.episode_states, self.episode_actions):
            self.action_state_visits[int(action), state.player_value -
                                     1, state.dealer_card-1] += 1

        #  update the value functions and the counters once episode complete
        # once the episode has completed and you know your reward, you use that reward to update all of the states
        # you saw in that episode with that reward
        for state, action in zip(self.episode_states, self.episode_actions):
            # find the location of the state in the matrix
            # use the equation for q on slide 16 in his lecture 5 (model free control) to do the update
            # this is also where the alpha step size comes in (alpha = 1/n(st, at)). as our discounting factor is 1,
            # the return (g) is just the reward from the previous action
            # q(st,at) = q(st, at) + [1/n(st, at)*(reward - q(st,at))]
            step_size = 1 / self.action_state_visits[int(action),
                                                     state.player_value-1, state.dealer_card-1]  # , make it 1 big 3d array, 3 layered, should be faster
            prev_value_func = self.value_function[int(action),
                                                  state.player_value-1, state.dealer_card-1]

            new_value_func = prev_value_func + \
                (step_size*(self.environment.player_reward - prev_value_func))

            self.value_function[int(action), state.player_value -
                                1, state.dealer_card-1] = new_value_func

        self.episode_states = []
        self.episode_actions = []

    def choose_action(self, state) -> bool:
        # we need to -1 each time to find the index
     #       hit_value, stick_value = self.value_function[:,
      #                                                   state.player_value-1, state.dealer_card-1]
        hit_value = self.value_function[0,
                                        state.player_value-1, state.dealer_card-1]
        stick_value = self.value_function[1,
                                          state.player_value-1, state.dealer_card-1]
        # each time you run the step, you need to find the state and action pair in the matrix, and determine which action
        # to take based on which state has a higher value function
        # if they are both even, you choose an action at random
        if hit_value == stick_value:
            # return not(np.random.uniform(0,1) > 0,5)
            # return not random.getrandbits(1)  # that was the original
            return bool(np.random.randint(0, 2))
        # with probability 1- epsilon, choose the greedy action
        epsilon = N0 / \
            (N0 + self.state_visits[0, state.player_value -
                                    1, state.dealer_card-1])
        # if random number is bigger than epsilon, choose the greedy action - this means we start
        # off exploratively, then become more greedy over time if number > epsilon:
        # here we choose the most valuable of the two actions using XOR gate
        return (np.random.uniform(0, 1) > epsilon) ^ (hit_value > stick_value)


if __name__ == '__main__':
    iterations = 100000  # was 10000000
    agent = Agent()
    agent.run_optimisation(iterations)
    plot_value_function(agent, iterations)
