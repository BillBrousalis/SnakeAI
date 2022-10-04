#!/usr/bin/env python3
import os, sys
import time
import torch
from ai.model import SimpleNet, SimpleNet2
from game.gameenv import GameEnv
from utilities import state_util, model_util, vis_util

def test(model, batch=False):
  env = GameEnv(mode='test')
  vis_args = vis_util.setup(env, title='Snake SimpleNet TESTING')
  # game-loop
  while env.isrunning:
    if not batch: time.sleep(0.05)
    state = env.getstate()
    vis_util.render(*vis_args, *state)
    state = state_util.state2vector(*env.getstate())
    state = torch.tensor(state, dtype=torch.float)
    action = [0, 0, 0, 0]
    action[torch.argmax(model(state)).item()] = 1
    env.step(action)
  print(f'[*] AI Score: {env.score}')
  return env.score

def batch(model, n):
  scores = []
  for i in range(n): scores.append(test(model, batch=True))
  print(f'Mean score after {n} games: {sum(scores)/n:.1f}')
  print(f'Best score after {n} games: {max(scores)}')
  
  
if __name__ == '__main__':
  net_fname = 'simplenet33.pth'
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cpu'): print(f'[ Testing on: CPU ]\n')
  else: print(f'[ Testing on: GPU ({torch.cuda.get_device_name(0)}) ]\n')
  model = SimpleNet(8, 4)
  #model = SimpleNet2(8, 4)
  model = model_util.load(model, 'models', net_fname).to(device)
  if len(sys.argv) >= 2: batch(model, int(sys.argv[1]))
  else: test(model)
