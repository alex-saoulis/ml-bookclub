from mountain_car import MountainCar
from actor_critic import ActorCriticAgent

from plotting import plot_curve

env = MountainCar()
acAgent = ActorCriticAgent(env, lambdaIn = 0.99)

print(acAgent.generateFeatureVector([-1, -1]))
acAgent.loadWeights("./weights/debug.pkl")
acAgent.train(numSteps=20000)

env.render("./trained_agent.gif","gif")
plot_curve(acAgent.historicalReward, filepath="./reward.png",
                       x_label="Episode", y_label="Reward",
                       x_range=(0, len(acAgent.historicalReward)), y_range=(-1.1,1.1),
                       color="red", kernel_size=500,
                       alpha=0.4, grid=True)

acAgent.saveWeights("./weights/debug.pkl")