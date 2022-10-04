#!/usr/bin/env python3
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Agent(nn.Module):
  def __init__(self, net, rmem, batch_size, lr):
    super(Agent, self).__init__()
    self.net = net
    self.rmem = rmem
    self.batch_size = batch_size
    self.lr = lr
    self.criterion = nn.MSELoss()
    self.optimizer = optim.Adam(net.parameters(), lr=self.lr)
    self.epsilon = 0
    self.gamma = 0.9

  def learn(self, state, action, reward, next_state, done):
    state = torch.tensor(np.array(state), dtype=torch.float)
    action = torch.tensor(np.array(action), dtype=torch.float)
    reward = torch.tensor(np.array(reward), dtype=torch.float)
    next_state = torch.tensor(np.array(next_state), dtype=torch.float)
    if len(state.shape) == 1:
      state = torch.unsqueeze(state, 0)
      next_state = torch.unsqueeze(next_state, 0)
      action = torch.unsqueeze(action, 0)
      reward = torch.unsqueeze(reward, 0)
      done = (done, )
    prediction = self.net(state)
    target = prediction.clone()
    for i in range(len(done)):
      Qnew = reward[i]
      if not done[i]:
        Qnew = reward[i] + self.gamma * torch.max(self.net(next_state[i]))
      target[i][torch.argmax(action[i]).item()] = Qnew
    self.optimizer.zero_grad()
    loss = self.criterion(target, prediction)
    loss.backward()
    self.optimizer.step()
    return Qnew, loss

  def learn_long_mem(self):
    if len(self.rmem) > self.batch_size:
      mini_sample = self.rmem.sample(self.batch_size)
    else:
      mini_sample = self.rmem.memory
    states, actions, rewards, next_states, dones = zip(*mini_sample)
    self.learn(states, actions, rewards, next_states, dones)
  
  def getaction(self, state, ngames):
    self.epsilon = 80 - (ngames*0.5)
    action = [0, 0, 0, 0]
    if random.randint(0, 200) < self.epsilon:
      action[random.randint(0, 3)] = 1
    else:
      state_tensor = torch.tensor(state, dtype=torch.float)
      prediction = self.net(state_tensor)
      action[torch.argmax(prediction).item()] = 1
    return action

  def cache(self, state, action, reward, next_state, done):
    self.rmem.push((state, action, reward, next_state, done))
