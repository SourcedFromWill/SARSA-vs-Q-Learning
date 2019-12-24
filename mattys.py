import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class GridWorld:
    def __init__(self, dim, start, finish, cliff):
        self.nY = dim[0]
        self.nX = dim[1]
        self.start = start
        self.finish = finish
        self.cliff = cliff
        
        # Create grid for the world
        self.world = [[0 for x in range(self.nX)] for y in range(self.nY)]
        
    def initialise(self):
        for y in range(self.nY):
            for x in range(self.nX):
                coords = (x, y)
                if coords == start:
                    self.world[y][x] = "S"
                elif coords == finish:
                    self.world[y][x] = "T"
                elif coords in cliff:
                    self.world[y][x] = "C"
                else:
                    self.world[y][x] = "o"
    
    def update(self, new_pos, old_pos, marker):
        self.world[old_pos[1]][old_pos[0]] = marker
        self.world[new_pos[1]][new_pos[0]] = "x"       

    def printWorld(self):
        for y in range(self.nY):
            for x in range(self.nX):
                print(self.world[y][x], end=" ")
            print()
        print()
        
class Agent():
    def __init__(self, dimensions, start, finish, cliff):
        # Initialise gridworld and set position
        self.GridWorld = GridWorld(dimensions, start, finish, cliff)
        self.GridWorld.initialise()
        
        # Agent coordinates
        self.x = start[0]
        self.y = start[1]
        self.pos = (self.x,self.y)
        
        # Agent moves and rewards
        self.moves = {0: "^", 1: "v", 2: "<", 3: ">"}
        self.reward = 0
    
    def reset(self):
        self.GridWorld.initialise()
        self.pos = self.GridWorld.start
        self.x = self.GridWorld.start[0]
        self.y = self.GridWorld.start[1]
    
    def get_reward(self):
        if self.GridWorld.world[self.y][self.x] == "C":
            reward = -100
        else:
            reward = -1
        return reward

    def move(self, action):
        cur_state = self.pos
        
        # Determine new (x,y) based on action
        if (action == 0):
            if (self.y > 0):
                self.y -= 1
        elif (action == 1):
            if (self.y < self.GridWorld.nY - 1):
                self.y += 1
        elif (action == 2):
            if (self.x > 0):
                self.x -= 1
        else:
            if (self.x < self.GridWorld.nX - 1):
                self.x += 1

        # Update the position of the agent
        self.pos = (self.x,self.y)
        new_state = self.pos

        # Get the reward for this movement
        reward = self.get_reward()

        # Check that new position isn't a cliff,
        # if it is we can reset the world
        if self.GridWorld.world[self.y][self.x] == "C":
            self.reset()
        else:
            # Update the gridworld
            self.GridWorld.update(new_state, cur_state, self.moves[action])
        
        return reward, cur_state, new_state

class ReinforcementLearning():
    def __init__(self, dimensions, start, finish, cliff, epsilon, alpha, gamma):
        self.Agent = Agent(dimensions, start, finish, cliff)
        self.episodes = 0
        
        # Learning rates
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        # Q-Values: qValues[state][action]
        self.qValues = defaultdict(lambda: defaultdict(lambda: 0))

    def get_qvalue(self, state, action):
        return self.qValues[state][action]

    def set_qvalue(self, state, action, Q):
        self.qValues[state][action] = Q
                  
    def max_state_value(self, state):
        values = []
        for action in range(4):
            values.append(self.get_qvalue(state, action))
        max_state_value = np.max(values)
        return max_state_value
        
    def update_qvalue(self, mode, state, action, reward, next_state=None):
        if mode == 'q':
            Q = self.get_qvalue(state, action)
            Q = Q + self.alpha * (reward + (self.gamma * self.max_state_value(next_state)) - Q)
            self.set_qvalue(state, action, Q)
        elif mode == 's':
            Q = self.get_qvalue(state, action)
            Q = Q + (reward + (self.gamma * Q) - Q)
            self.set_qvalue(state, action, Q)   
            
    def get_best_action(self, state):
        qvalues = []
        for action in range(4):
            qvalues.append(self.get_qvalue(state, action))
        best_action = np.argmax(qvalues)
        return best_action
   
    def get_action(self, state):
        if (np.random.rand() < self.epsilon):
            action = np.random.randint(4)
        else:
            action = self.get_best_action(state)
        return action
    
    def q_learning(self, episodes):
        episode_rewards = []
        while(self.episodes <= episodes):
            print("EPISODE: " + str(self.episodes))
            # Determine action
            action = self.get_action(self.Agent.pos)
            
            # Move agent
            reward, cur_state, new_state = self.Agent.move(action)
            self.Agent.reward += reward
            print("Reward for action: " + str(reward))
            print("Cumulative reward: " + str(self.Agent.reward))
            
            # Update q values
            self.update_qvalue('q', cur_state, action, reward, new_state)
            
            # Print out the world
            self.Agent.GridWorld.printWorld()
            
           # time.sleep(.5)
            
            # Check if agent reached terminal state
            if self.Agent.pos == self.Agent.GridWorld.finish:
                episode_rewards.append(self.Agent.reward)
                self.Agent.reset()
                self.Agent.reward = 0
                self.episodes += 1
                #time.sleep(2)
                
        return episode_rewards
    
    def sarsa(self, episodes):
        episode_rewards = []
        while(self.episodes <= episodes):
            print("EPISODE: " + str(self.episodes))
            # Determine action
            action = self.get_action(self.Agent.pos)
            
            # Move agent
            reward, cur_state, new_state = self.Agent.move(action)
            self.Agent.reward += reward
            print("Reward for action: " + str(reward))
            print("Cumulative reward: " + str(self.Agent.reward))
            
            # Update q values
            self.update_qvalue('s', cur_state, action, reward)
            
            # Print out the world
            self.Agent.GridWorld.printWorld()
            
           # time.sleep(.5)
            
            # Check if agent reached terminal state
            if self.Agent.pos == self.Agent.GridWorld.finish:
                episode_rewards.append(self.Agent.reward)
                self.Agent.reset()
                self.Agent.reward = 0
                self.episodes += 1
                time.sleep(2)
                
        return episode_rewards

if __name__ == "__main__":
    dimensions = (4,12) # Set the dimension of the world (rows, columns)
    start = (0,3) # Coordinates of the start point
    finish = (11,3) # Coordinates of the finish point
    cliff = [(i, 3) for i in range(11)] # Coordinates for the cliff
    gamma = 1
    epsilon = .1
    alpha = .1
    
    # Instantiate the reinforcement learning class
    # rl = ReinforcementLearning(dimensions, start, finish, cliff, epsilon, gamma, alpha)
    
    # Run Q Learning algorithm for n episodes
    # q_episode_rewards = rl.q_learning(500)
    
    print("STARTING SARSA")
    # time.sleep(10)
    
    rl = ReinforcementLearning(dimensions, start, finish, cliff, epsilon, gamma, alpha)
    
    # SARSA
    s_episode_rewards = rl.sarsa(500)
    
    # Calculate 10-point running average
    def moving_average(x, N):
        '''
        Moving avg function by Alleo
        https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
        '''
        
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    
    # Q Learning
    # q_mavg = moving_average(q_episode_rewards, 10)
    s_mavg = moving_average(s_episode_rewards, 10)
    
    # Plot graphs
    plt.figure()
    # plt.plot(range(len(q_mavg)), q_mavg, label="Q Learning")
    plt.plot(range(len(s_mavg)), s_mavg, label="SARSA")
    # plt.ylim(ymin=-750)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend(loc="lower right", prop={'size': 10})
    leg = plt.legend()
    for line in leg.get_lines():
        line.set_linewidth(1)
    plt.show()