from actor_critic import ActorCriticAgent
from mountain_car import MountainCar
from plotting import plot_curve


if __name__ == '__main__':
    # define the inputs
    env = MountainCar()

    ac_agent = ActorCriticAgent(environment=env, action_spread=0.8, lam=0.99)
    ac_agent.train(n_iterations=20000)
    # env.render("./trained_agent.gif","gif")
    plot_curve(ac_agent.historical_reward, filepath="./reward.png",
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(ac_agent.historical_reward)), y_range=(-1.1,1.1),
                       color="red", kernel_size=500,
                       alpha=0.4, grid=True)

    # state = [-0.9, -1]  # [position, velocity]
    # print(ac_agent.generate_feature_vector(state))