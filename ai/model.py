#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
  def __init__(self, nin, nout):
    super().__init__()
    sz = 256
    self.linear1 = nn.Linear(nin, sz)
    self.linear2 = nn.Linear(sz, nout)

  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    return x

class SimpleNet2(nn.Module):
  def __init__(self, nin, nout):
    super().__init__()
    sz = 64
    self.linear1 = nn.Linear(nin, sz)
    self.linear2 = nn.Linear(sz, nout)

  def forward(self, x):
    x = self.linear1(x)
    x = self.linear2(x)
    return x
