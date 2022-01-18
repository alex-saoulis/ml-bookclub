
from CartpoleDriver import CartpoleDriver
import gym
MINIMAL_ROUNDS = 100 # the description says env terminates when there are 200 rounds
env = gym.make("CartPole-v1")
observation = env.reset()
driver = CartpoleDriver()


def run_episode(env, driver):
  """[summary]

  Args:
      env ([type]): [description]
      driver ([type]): [description]

  Returns:
      [type]: [description]
  """
  done = False
  while not done:
    env.render()
    action = driver.decide(env)
    observation, reward, done, info = env.step(action)
    driver.update_position(observation, reward, done, action)
  observation = env.reset()
  return reward,info

for _ in range(MINIMAL_ROUNDS):
  reward, info = run_episode(env, driver)
  print("failed at: ", reward )

print("solved?: ", driver.check_if_solved())
env.close()