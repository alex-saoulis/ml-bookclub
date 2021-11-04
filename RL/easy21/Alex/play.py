from easy21 import Easy21, State, Card, Colour, Action

class DumbAI:

    def playAI(self, env : Easy21):
        state = State(env.drawBlackCard().getValue(), env.drawBlackCard().getValue())
        while True:
            action = self.chooseAction(state)
            state = env.step(state, action)
            if state.terminal:
                print(f"Player score: {state.playerScore}")
                print(f"Game over: reward = {state.reward}")
                break
            

    def chooseAction(self, state: State):

        if state.playerScore < 14:
            return Action.HIT
        else:
            return Action.STICK

env = Easy21(verbose=True)
ai = DumbAI()

ai.playAI(env)