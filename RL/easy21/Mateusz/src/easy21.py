import numpy as np
import random

class easy21_game:

    def __init__(self, aces = False):

        self.term = False

        self.player_score = 0
        self.dealer_score = 0
        
        self.aces = aces
        self._cards_values = list(range(1,11,1))


        #### trigger warning! lazy enums! ####

        self._colors = ["black","red","black"]
        self._color_mod = {"black":1,"red":-1}
        
        self.action_space = ["hit","stick"]
        self.action_enum = {"hit":1,"stick":0}

        self._states = ["player_bust","dealer_bust","draw","player_turn","dealer_turn","player_win","dealer_win"]

        self.state = "player_turn"

        self.reset()



    def pick_card(self,pos = False):

        if pos:
            return random.choice(self._cards_values)*1
        else:
            return random.choice(self._cards_values)*self._color_mod[random.choice(self._colors)]



    def step(self,action):

        done = False

        if self.state == "player_turn":

            if self.action_enum[action]  == 1:

                add_score = self.pick_card()

                if np.abs(add_score == 1) and self.aces:
                    if (self.player_score + 11) > 21:
                        add_score = 1
                    else:
                        add_score = 11
                
                self.player_score += add_score

                if self.player_score > 21 or self.player_score < 1:
                    self.state = "dealer_win"
                    reward = -1
                    done = True

                else:
                    reward = 0

                info = {"player":self.player_score, "dealer":self.dealer_score,"state":self.state}
            

            if self.action_enum[action]  == 0:

                while self.dealer_score < 17 and self.dealer_score >= 1:


                    add_score = self.pick_card()

                    if add_score == 1 and self.aces:
                        if (self.dealer_score + 11) > 21:
                            add_score = 1
                        else:
                            add_score = 11

                    self.dealer_score += add_score


                    
                if self.dealer_score > 21 or self.dealer_score < 1:

                    self.state = "player_win" 
                    reward = 1

                elif self.dealer_score == self.player_score:

                    self.state = "draw"
                    reward = 0

                elif self.dealer_score < self.player_score:

                    self.state = "player_win"
                    reward = 1

                elif self.dealer_score > self.player_score:

                    self.state = "dealer_win"
                    reward = -1
                

                done = True 

                info = {"player":self.player_score, "dealer":self.dealer_score,"state":self.state}


        observation = (self.player_score,self.dealer_score)


        # if self.state == "player_win":
        #     print(self.player_score, self.dealer_score)

        if done:
            self.term = True

        return observation , reward , done, info
    
    def reset(self):

        self.state = "player_turn"
        self.term = False

        self.player_turn = 0
        self.dealer_turn = 0

        self.player_score = self.pick_card(pos = True)
        self.dealer_score = self.pick_card(pos = True)
        
        return True

    def get_state(self,human = False):

        if not human:
            return self.player_score,self.dealer_score
        else:
            return {"player":self.player_score, "dealer":self.dealer_score,"state":self.state}