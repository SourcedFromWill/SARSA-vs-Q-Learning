import numpy as np
import math
import sys
import random
from collections import deque
#import matplotlib as mpl
#from matplotlib import pyplot
import os
#import pandas as pd
#import matplotlib.pyplot as plt
#from matplotlib.colors import hsv_to_rgb
import time



class GridModel():
    def __init__(self, id):
        """ Initialise the cliffwalking gridworld """
        # Load in the grid model from file
        with open('grid.txt', 'r') as grid:
            stateTypes = [str(grid.readline()) for a in range(6) ]
        self.agent = id
        # Define states
        self.states = ( [[ State(a, b, stateTypes[a][b]) for a in range(6) ] for b in range(14) ] )
            
    def printGrid(self, x, y):
        """ Print the grid onto a terminal"""
        for b in range(4):
            print()
            for a in range(12):
                if (a+1 == x and b+1 == y):
                    print(self.agent, end=" ")
                else:
                    print(self.states[a+1][b+1].stateType, end=" ")

        print()
    
        '''def _draw_grid(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 4))
        self.ax.grid(which='minor')       
        self.q_texts = [self.ax.text(*self._id_to_position(i)[::-1], '0',
                                     fontsize=11, verticalalignment='center', 
                                     horizontalalignment='center') for i in range(12 * 4)]     
         
        self.im = self.ax.imshow(hsv_to_rgb(self.grid), cmap='terrain',
                                 interpolation='nearest', vmin=0, vmax=1) 
                
    '''
class State(): 
    def __init__(self, x, y, stateType):
        """ Represents a state with 4 actions """
        self.xy = (x, y)
        # action rewards in order 0 up, 1 down, 2 left, 3 right
        self.qValues = {0: 0, 1: 0, 2: 0, 3: 0}
        self.stateType = stateType

        # set rewards for states according to their types read in from the txt file
        if (self.stateType == 'C'):
            self.reward = -100
        elif (self.stateType == 'G'):
            self.reward = 10
        else:
            self.reward = -1

    def getQ(self, action):        return self.qValues[action]

    def setQ(self, action, Q):
        self.qValues.update({action: Q})

    def maxQValue(self):
        """ Returns key, value pair of the paximum action """
        v=list(self.qValues.values())
        k=list(self.qValues.keys())
        return  k[np.argmax(v)], v[np.argmax(v)]


class Agent():
    def __init__(self, id):
        """ Agent for SARSA or Qlearning in  the Cliffworld setting"""

        # Grid variables
        self.x = 1
        self.y = 4
        self.minX = 1
        self.maxX = 12 
        self.minY = 1
        self.maxY = 4
        self.actionSize = 4
        self.model = GridModel(id)

        # Learning variables
        self.alpha = .1
        self.gamma = 1
        self.epsilon = .1

        # Stats
        self.rewards = []
        self.reward = 0
        self.episode = 0

    def qIteration(self):
        """ Q learning for each step of episode """
        
        # Get greedy action for exploitation
        action, maxQ = self.model.states[self.x][self.y].maxQValue()
        oldx = self.x
        oldy = self.y

        # Random Chance of exploring in a random direction
        if random.random() <= self.epsilon:
            action = random.randrange(self.actionSize)

        self.move(action)

        chosenX = self.x
        chosenY = self.y

        self.correctPosition(action)

        actionPrime, maxQPrime = self.model.states[self.x][self.y].maxQValue()

        # Find Q and R values for the update formula
        Q = self.model.states[oldx][oldy].getQ(action)
        R = self.model.states[chosenX][chosenY].reward

        # Update new Q value into state
        actionQ = Q + self.alpha * ( R + (self.gamma * maxQPrime) - Q)
        self.model.states[oldx][oldy].setQ(action, actionQ) 

        #print(Q, R, self.model.states[oldx][oldy].getQ(action))
        print(self.model.states[self.x][self.y].qValues, action) 
        self.model.printGrid(self.x, self.y)
        self.reward += R
        

    def move(self, action):
        # Move to new state
        
        if action == 0:
            self.y = self.y-1
        elif action == 1:
            self.y = self.y+1
        elif action == 2:
            self.x = self.x-1
        elif action == 3:
            self.x = self.x+1

    def correctPosition(self, action):
            
        # If encountered a wall, move back
        if self.model.states[self.x][self.y].stateType == "X":
            if action == 0:
                self.y = self.y+1
            elif action == 1:
                self.y = self.y-1
            elif action == 2:
                self.x = self.x+1
            elif action == 3:
                self.x = self.x-1

        # If encountered a cliff, move to start
        if self.model.states[self.x][self.y].stateType == "C":
            self.x = 1
            self.y = 4
        
            #self.episode += 1
            #self.rewards.append(self.reward)
            #self.reward = 0

        # If encountered the Goal, move to start
        if self.model.states[self.x][self.y].stateType == "G":
            self.x = 1
            self.y = 4

            print("GOAL REACHED!")
            time.sleep(.1)
            self.episode += 1
            self.rewards.append(self.reward)
            self.reward = 0

    
    def sarsaIteration(self):
        """ SARSA learning episode loop """
        # Get greedy action for exploitation
        action, maxQ = self.model.states[self.x][self.y].maxQValue()
        oldx = self.x
        oldy = self.y

        # Random Chance of exploring in a random direction
        if random.random() <= self.epsilon:
            action = random.randrange(self.actionSize)

        self.move(action)

        chosenX = self.x
        chosenY = self.y

        self.correctPosition(action)

        actionPrime, maxQPrime = self.model.states[self.x][self.y].maxQValue()

        # Find Q and R values for the update formula
        Q = self.model.states[oldx][oldy].getQ(action)
        R = self.model.states[chosenX][chosenY].reward

        # Update new Q value into state
        actionQ = Q + ( R + (self.gamma * Q) - Q)
        self.model.states[oldx][oldy].setQ(action, actionQ) 

        print(self.model.states[self.x][self.y].qValues, action) 
        self.model.printGrid(self.x, self.y)
        self.reward += R

        
def main():
    sAgent = Agent('s')
    qAgent = Agent('q')

    # Run 100 episodes
    while (qAgent.episode <= 100):
        sAgent.sarsaIteration()
        print("Episode: " + str(qAgent.episode) + ", Reward: " + str(sAgent.reward) )
        
        qAgent.qIteration()
        print("Episode: " + str(qAgent.episode) + ", Reward: " + str(qAgent.reward) )

    


main()