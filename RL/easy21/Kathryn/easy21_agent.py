import numpy as np
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from easy21_environment import Easy21Env, Action


def plot_value_function(episode, n_iterations):
    nx, ny = 10, 21
    x = range(nx)
    y = range(ny)

    data = np.maximum(episode.value_function[0], episode.value_function[1])

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    x, y = np.meshgrid(x, y)
    ha.plot_surface(x, y, data)

    ha.set_title(f'value function with {n_iterations} iterations')
    ha.set_xlabel('dealer first card')
    ha.set_ylabel('player score')
    ha.set_zlabel('value')

    plt.savefig("trial")
    plt.show()


class Agent():
    def __init__(self):
        self.environment = Easy21Env()
        self.reset_states_and_actions()

        self.n0 = 1000
        # the number of times a state is visited (2d array)
        self.ns = np.zeros((1, 21, 10))
        # the number of times an action is used in a state (3d array)
        self.nsa = np.zeros((2, 21, 10))

        # the policy is implicit in this
        self.value_function = np.zeros((2, 21, 10))
        # this should be a 3d array where the x axis is the player score, the y axis is the dealer score and the
        # z axis is the action (hit or stick). then, for a given player and dealer score and action, (which you find
        # by navigating through using the indices), you go with the value / action that gives the highest reward
        # the counter matrix should look the same as the other one, but rather than updating with the reward each time,
        # we increment the state-action pairs with the number of times we saw that state in the episode by 1

    def run_episode(self, episode_number):
        self.environment.reset()
        # each time you run the step, you need to find the state and action pair in the matrix, and determine which action
        # to take based on which state has a higher value function
        # if they are both even, you choose an action at random
        self.episode_states.append(self.environment.state)
        while self.environment.done is not True:
            action = self.choose_action(self.environment.state)
            new_state = self.environment.step(action)

            # store the values for updates later
            self.episode_actions.append(action)
            if self.environment.done is not True:
                self.episode_states.append(new_state)

        # one we've completed the episode, we need to update the value functions and the counters
        self.update_ns_counter()
        self.update_nsa_counter()
        self.update_value_function()

        self.reset_states_and_actions()
        # return self.environment.state, self.environment.player_reward

    def reset_states_and_actions(self):
        self.environment.done = False
        self.episode_states = []
        self.episode_actions = []

    def choose_action(self, state):
        # we need to -1 each time to find the index
        hit_value = self.value_function[0,
                                        state.player_value-1, state.dealer_card-1]
        stick_value = self.value_function[1,
                                          state.player_value-1, state.dealer_card-1]

        if hit_value == stick_value:
            action_value = np.random.randint(0, 2)
        else:
            # with probability 1- epsilon, choose the greedy action
            epsilon = self.n0 / \
                (self.n0 + self.ns[0, state.player_value -
                                   1, state.dealer_card-1])
            # if random number is bigger than epsilon, choose the greedy action - this means we start
            # off exploratively, then become more greedy over time
            number = np.random.uniform(0, 1)
            if number > epsilon:
                # here we choose the most valuable of the two actions
                if hit_value > stick_value:
                    action_value = 0
                else:
                    action_value = 1
            else:
                # here, we pick the less valuable of the two
                if hit_value < stick_value:
                    action_value = 0
                else:
                    action_value = 1

        if action_value == 0:
            action = Action.HIT
        else:
            action = Action.STICK
        return action

    def update_value_function(self):
        # once the episode has completed and you know your reward, you use that reward to update all of the states
        # you saw in that episode with that reward
        for state, action in zip(self.episode_states, self.episode_actions):
            # find the location of the state in the matrix
            # use the equation for q on slide 16 in his lecture 5 (model free control) to do the update
            # this is also where the alpha step size comes in (alpha = 1/n(st, at)). as our discounting factor is 1,
            # the return (g) is just the reward from the previous action
            # q(st,at) = q(st, at) + [1/n(st, at)*(reward - q(st,at))]
            step_size = 1 / self.nsa[action.value,
                                     state.player_value-1, state.dealer_card-1]
            prev_value_func = self.value_function[action.value,
                                                  state.player_value-1, state.dealer_card-1]

            new_value_func = prev_value_func + \
                (step_size*(self.environment.player_reward - prev_value_func))

            self.value_function[action.value, state.player_value -
                                1, state.dealer_card-1] = new_value_func

    def update_ns_counter(self):
        for state in self.episode_states:
            self.ns[0, state.player_value-1, state.dealer_card-1] += 1

    def update_nsa_counter(self):
        for state, action in zip(self.episode_states, self.episode_actions):
            self.nsa[action.value, state.player_value -
                     1, state.dealer_card-1] += 1

    def run_optimisation(self, n_iterations):
        for i in tqdm.tqdm(range(n_iterations)):
            self.run_episode(i)


if __name__ == '__main__':
    n_iterations = 10000000
    agent = Agent()
    agent.run_optimisation(n_iterations)
    plot_value_function(agent, n_iterations)
