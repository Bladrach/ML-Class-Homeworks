import numpy as np
import random
class QMap():
    def __init__(self):
        self.height = 10
        self.width = 10
        self.posX = 0
        self.posY = 0
        self.endX = self.width-1
        self.endY = self.height-1
        # impassible mountains
        self.mountain1X = 3; self.mountain1Y = 3
        self.mountain2X = 5; self.mountain2Y = 6
        self.mountain3X = 8; self.mountain3Y = 2
        self.mountain4X = 4; self.mountain4Y = 7
        self.mountain5X = 9; self.mountain5Y = 4
        self.mountain6X = 1; self.mountain6Y = 1
        # toxic mists
        self.toxic1X = 7; self.toxic1Y = 5
        self.toxic2X = 2; self.toxic2Y = 8
        self.toxic3X = 5; self.toxic3Y = 0
        self.TM = 0
        self.actions = [0, 1, 2, 3]
        self.stateCount = self.height*self.width
        self.actionCount = len(self.actions)

    def reset(self):
        self.posX = 0
        self.posY = 0
        self.done = False
        self.TM = 0
        return 0, 0, False, 0

    # take action
    def step(self, action):
        
        if action==0: # left
            self.posX = self.posX-1 if self.posX>0 and (self.posX-1 != self.mountain1X or self.posY != self.mountain1Y) \
                                            and (self.posX-1 != self.mountain2X or self.posY != self.mountain2Y) \
                                            and (self.posX-1 != self.mountain3X or self.posY != self.mountain3Y) \
                                            and (self.posX-1 != self.mountain4X or self.posY != self.mountain4Y) \
                                            and (self.posX-1 != self.mountain5X or self.posY != self.mountain5Y) \
                                            and (self.posX-1 != self.mountain6X or self.posY != self.mountain6Y) else self.posX
        if action==1: # right
            self.posX = self.posX+1 if self.posX<self.width-1 and (self.posX+1 != self.mountain1X or self.posY != self.mountain1Y) \
                                            and (self.posX+1 != self.mountain2X or self.posY != self.mountain2Y) \
                                            and (self.posX+1 != self.mountain3X or self.posY != self.mountain3Y) \
                                            and (self.posX+1 != self.mountain4X or self.posY != self.mountain4Y) \
                                            and (self.posX+1 != self.mountain5X or self.posY != self.mountain5Y) \
                                            and (self.posX+1 != self.mountain6X or self.posY != self.mountain6Y) else self.posX
        if action==2: # up
            self.posY = self.posY-1 if self.posY>0 and (self.posY-1 != self.mountain1Y or self.posX != self.mountain1X) \
                                            and (self.posY-1 != self.mountain2Y or self.posX != self.mountain2X) \
                                            and (self.posY-1 != self.mountain3Y or self.posX != self.mountain3X) \
                                            and (self.posY-1 != self.mountain4Y or self.posX != self.mountain4X) \
                                            and (self.posY-1 != self.mountain5Y or self.posX != self.mountain5X) \
                                            and (self.posY-1 != self.mountain6Y or self.posX != self.mountain6X) else self.posY
        if action==3: # down
            self.posY = self.posY+1 if self.posY<self.height-1 and (self.posY+1 != self.mountain1Y or self.posX != self.mountain1X) \
                                            and (self.posY+1 != self.mountain2Y or self.posX != self.mountain2X) \
                                            and (self.posY+1 != self.mountain3Y or self.posX != self.mountain3X) \
                                            and (self.posY+1 != self.mountain4Y or self.posX != self.mountain4X) \
                                            and (self.posY+1 != self.mountain5Y or self.posX != self.mountain5X) \
                                            and (self.posY+1 != self.mountain6Y or self.posX != self.mountain6X) else self.posY

        toxicMists = self.posX==self.toxic1X and self.posY==self.toxic1Y \
                    or self.posX==self.toxic2X and self.posY==self.toxic2Y\
                    or self.posX==self.toxic3X and self.posY==self.toxic3Y
        done = self.posX==self.endX and self.posY==self.endY
        # mapping (x,y) position to number between 0 and 10x10-1=99
        nextState = self.width*self.posY + self.posX
        if done:
            reward = 100
            TM = 0
        elif toxicMists:
            reward = -100
            TM = 1
        else:
            reward = -1
            TM = 0
        return nextState, reward, done, TM

    # return a random action
    def randomAction(self):
        return np.random.choice(self.actions)

    # display quest map
    def render(self):
        for i in range(self.height):
            for j in range(self.width):
                if self.posY==i and self.posX==j:
                    print("W", end='')
                elif self.endY==i and self.endX==j:
                    print("B", end='')
                elif self.mountain1Y==i and self.mountain1X==j \
                    or self.mountain2Y==i and self.mountain2X==j \
                    or self.mountain3Y==i and self.mountain3X==j \
                    or self.mountain4Y==i and self.mountain4X==j\
                    or self.mountain5Y==i and self.mountain5X==j\
                    or self.mountain6Y==i and self.mountain6X==j:
                    print("□", end='')
                elif self.toxic1Y==i and self.toxic1X==j \
                    or self.toxic2Y==i and self.toxic2X==j \
                    or self.toxic3Y==i and self.toxic3X==j:
                    print("x", end='')
                else:
                    print(".", end='')
            print("")

    