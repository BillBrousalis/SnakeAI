#!/usr/bin/env python3
import random
from collections import namedtuple, deque

class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)
  
  def __len__(self):
    return len(self.memory)

  def push(self, t):
    self.memory.append(t)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)
