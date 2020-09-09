# -*- coding: utf-8 -*-
import torch as t
import torch.nn as nn
import numpy as np
import torch.nn.functional as f
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import os 
import random

class Network (nn.Module):
    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size= input_size
        self.nb_actions =nb_actions
        self.fc1= nn.Linear(self.input_size,30)
        self.fc2=nn.Linear(30, self.nb_actions)
        
    def forward(self, state):
        x =f.relu(self.fc1(state))# activation of the hidden neural network
        
        q_values= self.fc2(x)
        return q_values  
class ReplayingMemory(object):
    
    def __init__(self , capacity):
        self.capacity=capacity
        self.memory = []
    def push (self, event):
        self.memory.append(event)
        if len(self.memory)> self.capacity:
            del self.memory[0]
    def sample (self, batch_size):
        sample= zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(t.cat(x,0)),sample)
    
    
class Dqn():
    def __init__(self , input_size, nb_actions, gamma):
        self.gamma = gamma 
        self.reward_table = []
        self.model = Network(input_size, nb_actions)
        self.ReplayingMemory = ReplayingMemory(100000) 
        self.optimizer = optim.Adam(self.model.parameter(), lr =0.001)
        self.last_state= t.Tensor(input_size).unsqueeze(0)# input size= 5 left rigth -left -rigth -froward
        self.last_action = 0 ##action turn 20 degree or - 20 or 0
        self.last_reward= 0
    def choose_action( self, state):
        prob = f.softmax(self.model(Variable(state, volatile =True))*100)# t =100 si t est > la sertitude est >
        action = prob.multinomial()
        return action.data[0,0]
    def learn (self, batch_state, batch_next_state, batch_reward , batch_action):
        outputs= self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs= self.model(batch_next_state).detach().max(1)[0]
        target =self.gamma*next_outputs+batch_reward
        td_loss = f.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables=True)
        self.optimizer.step()# update the weigth of neural
    def update (self, reward, new_signal):
        new_state= t.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, t.Tensor([int(self.last_action)]),t.Tensor([self.last_reward])))
        action = self.choose_action(new_state)
        if len(self.memory.memory)> 100:
            batch_state, batch_next_state, batch_reward , batch_action= self.memory.sample(100)
        self.laern(batch_state, batch_next_state, batch_reward , batch_action)
        self.last_action= action 
        self.last_state= new_state
        self.last_reward= reward
        self.reward_table.append(reward)
        if len(self.reward_table)>1000:
            del self.reward_table[0]
        return action
    def score(self):
        return sum(self.reward_table)/(len(self.reward_table)+1)
    def save(self):
        t.save({'state_dict':self.model.state_dict(),
                 'optimizer':self.optimizer.state_dict() },
               'last_brain.pth')
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = t.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
"""
Created on Sat Aug 29 21:04:43 2020

@author: Mon pc
"""


