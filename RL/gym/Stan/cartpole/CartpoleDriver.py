# https://gym.openai.com/envs/CartPole-v1/
import numpy as np
SOLVED_AVERAGE = 195
SOLVED_CONSECUTIVE = 100
MAXMIUM_EPISODES = 200
FORGETTING_MODIFIER = 0.005

class CartpoleDriver():
    def __init__(self): 
        """
        need to have decreasing to the past multiplier, so that long runs have advantage
        also learning after every episode, not every step
        and it's about keeping it at that high level, nothing else
        which parameters do matter? idk, let's try with just velocity, it might work
        """
        print("starting the agent")
        self.value_function = np.array([])
        self.local_history= np.array([])
        self.reward_history = np.array([])
        self.episode_no = 1
    
    def decide(self, env):
        position, velocity, angle, angular_velocity = env.state
        # need to do that linspace
        # np.linspace(50, 50, num=10000)
        if self.value_function[velocity ,:,1] > self.value_function[velocity, :, -1]:
          return 1
        return -1
        # return env.action_space.sample()
    
    def get_n_periods_average(self, n_periods):
        return np.average(self.reward_history[-n_periods:])
    
    def check_if_solved(self):
        return self.get_n_periods_average(SOLVED_CONSECUTIVE) > SOLVED_AVERAGE

    def close_episode(self):
        self.value_function += self.local_history
        self.reset_reward_counter()
        self.local_history = np.array([])

    def update_position(self, observation, reward, done, action):
      if done:
        self.close_episode()
        self.episode_no+=1
      else:
        self.reward_history[self.episode_no, :-1] += reward
        self.local_history += [observation[2], reward, action] # getting just the speed
        self.local_history = self.local_history * [1, (1-FORGETTING_MODIFIER), 1]
