# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 14:05:20 2020

@author: Mon pc
"""
import gym 
import torch
import torch.nn as nn
import torch.optim as optima
import random
from collections import deque
import numpy as np


#####################'INItIALISATION###########
ENV_Name ='CartPole-v0'
Batch_size = 20
LEARNING_rate= 0.001
MAX_Explore =1.0
MIN_Explore  =0.1
EXPLORE_DECAY = 0.995
MEMORY_LEN =1_000_000
UPDATE_FREQ= 10
GAMMA=0.95


###########neural Network MOdel##############
def CartPoleModel(observation_space,action_space):
    return nn.Sequential(
        nn.Linear(observation_space, 24),
        nn.ReLU(),
        nn.Linear(24, 24),
        nn.ReLU(),
        nn.Linear(24, action_space)
        )

#################DQN Class#####
class DQN:
    def __init__(self,observation_space,action_space):
        self.exploration_rate=MAX_Explore
        self.action_space=action_space
        self.observation_space=observation_space
        self.memory = deque(maxlen=MEMORY_LEN)
        #define target net an policy 
        self.target_net = CartPoleModel(self.observation_space,self.action_space)
        self.policy = CartPoleModel(self.observation_space,self.action_space)
        """copy the wiegth of nodes """
        self.target_net.load_state_dict(self.policy.state_dict())
        self.target_net.eval()
        """define a lossfunction and optimizer"""
        self.critirion =nn.MSELoss()#loss function 
        self.optimizer=optima.Adam(self.policy.parameters())#optimization function 
        self.explore_limit=False # limit flag
        """looad memory function"""
        def load_momory (self, state,action,reward,next_state,terminal):
            self.memory.append(state,action,reward,next_state,terminal)
        """prediction action function"""
        def predict_action(self,state):
            random_nmber = np.random.rand()
            if random_nmber <self.exploration_rate:
                return random.randrange(self.action_space)
            q_values =self.target_net(state).detch().numpy()
            return np.argmax(q_values[0])
        def experience_replay(self):
            if len(self.memory)<Batch_size:
                return
            batch=random.sample(self.memory,Batch_size)
            for state,action,reward, next_state, terminal in batch:
                q_update=reward
                if not terminal:
                    q_update=reward+GAMMA
            self.target_net(next_state).max(axis=1)[0]
            q_values=self.target_net(state)
            q_values[0][action]=q_update
            """calculate the losss and  """
            loss=self.critirion(self.policy(state),q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            """update exporation rate """
            if not self.explore_limit:
                self.exploration_rate*=EXPLORE_DECAY
                if self.exploration_rate<MIN_Explore:
                    self.exploration_rate=MIN_Explore
                    self.explore_limit=True
                
            
        
        
        
        
