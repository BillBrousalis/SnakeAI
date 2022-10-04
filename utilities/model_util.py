#!/usr/bin/env python3
import os
import torch

def save(net, path, name):
  fname = os.path.join(path, name)
  torch.save(net.state_dict(), fname)

def load(net, path, name):
  fname = os.path.join(path, name)
  net.load_state_dict(torch.load(fname))
  net.eval()
  return net
