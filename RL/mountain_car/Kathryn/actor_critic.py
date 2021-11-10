import numpy as np
from tqdm import tqdm


"""
Critic updates weights w by linear TD(0)
Actor updates weights (theta) by policy gradient

Initialise s, θ
Sample a ∼ πθ
for each step do
    Sample reward r = Ras; sample transition s' ∼ Pas,·
    Sample action a' ∼ πθ(s', a')
    δ = r + γQw(s', a') − Qw (s, a)
    θ = θ + α∇θ log πθ(s, a)Qw (s, a)
    w ← w + βδφ(s, a)
    a ← a', s ← s'
end for
"""


class ActorCriticAgent():

    def __init__(self, environment, action_spread=0.5, gamma=1, alpha=0.01, lam=0.9):
        # action spread 
        self.environment = environment

        self.action_sigma = action_spread
        self.lam = lam
        self.gamma = gamma  # discount for future reward
        self.alpha = alpha  # step size

        self.init_weights()
        self.init_eligibility_traces()
        self.init_rbf_references()

    def init_weights(self):
        self.actor_weights = np.zeros((64))  # theta
        self.critic_weights = np.zeros((64))  # w

    def init_eligibility_traces(self):
        self.critic_eligibility = np.zeros((64))
        self.actor_eligibility = np.zeros((64))

    def init_rbf_references(self):
        rbf_xpositions = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4]
        rbf_velocities = np.linspace(-1.2, 1.2, 8)

        self.xpos_rbf_vector = np.concatenate(
            [[rbf_xpos for i in range(8)] for rbf_xpos in rbf_xpositions])
        self.vel_rbf_vector = np.concatenate(
            [[rbf_vel for rbf_vel in rbf_velocities] for i in range(8)])

    def generate_feature_vector(self, state):
        position, velocity = state

        sig_pos = 2*0.1**2
        xpos_vector = np.ones((64))*position
        vel_vector = np.ones((64))*velocity

        xpos_feature = np.exp(-((np.subtract(xpos_vector,
                                             self.xpos_rbf_vector)**2/(2*sig_pos))))
        vel_feature = np.exp(-((np.subtract(vel_vector,
                                            self.vel_rbf_vector)**2/(2*sig_pos))))
        feature_vector = np.multiply(xpos_feature, vel_feature)
        norm = np.linalg.norm(feature_vector)
        return feature_vector/norm

    def get_action(self, features):
        # evaluate the policy (actor):
        # we will use a Gaussian policy because we're in a continuous feature space so we
        # sample from the continuous action policy
        # continuous action a ~ N(theta*x, sigma**2)
        action_sample = np.random.normal(np.dot(features, self.actor_weights), self.action_sigma**2)
        return  min(max(action_sample,-0.2),0.2)

    def get_value_function(self, state):
        # like before, to get the value function (from the critic), we
        # find the dot product between the trained weights and the feature
        # vector
        features = self.generate_feature_vector(state)
        value_func = np.dot(self.critic_weights, features)
        return value_func

    def update_weights(self, new_state, old_state, action, reward):
        # w ← w + αδtet
        old_features = self.generate_feature_vector(old_state)

        error = self.calculate_error(reward, new_state, old_state)

        self.update_critic_eligibility(old_features)
        self.update_actor_eligibility(action, old_features)

        self.critic_weights += self.alpha * error * self.critic_eligibility
        self.actor_weights += self.alpha * error * self.actor_eligibility

    def calculate_error(self, reward, new_state, old_state):
        # δt = rt+1 + γV(st+1) − V(st)
        if self.done:
            new_value = 0
        else:
            new_value = self.get_value_function(new_state)

        old_value = self.get_value_function(old_state)
        error = reward + self.gamma*new_value - old_value
        return error

    def update_critic_eligibility(self, features):
        # features = self.generate_feature_vector(state)
        # et = γλet−1 + φ(st)      φ is feature vector
        self.critic_eligibility *= self.gamma*self.lam
        self.critic_eligibility += features

    def update_actor_eligibility(self, action, features):
        # et+1 = λet + ∇θ log πθ(s, a)
        # where ∇θ log πθ(s, a) = (a − µ(s))φ(s) / σ**2 (score function)
        score_func = self.get_score_function(action, features)
        self.actor_eligibility *= self.lam
        self.actor_eligibility += score_func

    def get_score_function(self, action, features):
        # score function = (a − µ(s))φ(s) / σ**2
        # µ(s) = φ(s).θ  # dot product of actor weights and feature for current
        # state
        mean_state = min(max(np.dot(features, self.actor_weights), -0.2), 0.2)
        score_func = ((action - mean_state)*features) / self.action_sigma ** 2
        return score_func

    def run_episode(self):
        total_reward = 0
        self.init_eligibility_traces()

        state = self.environment.reset()
        self.done = False

        while not self.done:

            features = self.generate_feature_vector(state)
            action = self.get_action(features)

            new_state, reward, self.done = self.environment.step(action)
            total_reward += reward
            
            self.update_weights(new_state, state, action, reward)
            
            state = new_state
            
        return total_reward

    def train(self, n_iterations):
        self.historical_reward = []
        for i in tqdm(range(n_iterations)):
            # if i == 10000:
            #     self.action_sigma = 0.2
            r = self.run_episode()
            self.historical_reward.append(r)

