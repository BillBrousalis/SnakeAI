#!/usr/bin/env python3
import time
import torch
from ai.agent import Agent
from ai.model import SimpleNet, SimpleNet2
from ai.replay_memory import ReplayMemory
from game.gameenv import GameEnv
from utilities import state_util, model_util, plot_util, vis_util

def train():
  SAVEPATH = './models'
  # cuda
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  if device == torch.device('cpu'): print(f'\n[ Training on: CPU ]\n')
  else: print(f'\n[ Training on: GPU ( {torch.cuda.get_device_name(0)} ) ]\n')
  # params
  BATCH_SIZE = 2048
  LR = 1e-3
  # instances
  env = GameEnv(mode='train')
  net = SimpleNet(8, 4).to(device)
  #net = SimpleNet2(8, 4).to(device)
  # put net in training mode
  net.train()
  # replay memory - agent
  rmem = ReplayMemory(100_000)
  agent = Agent(net, rmem, BATCH_SIZE, LR) 
  # data
  highscore = 0
  ngames = 0
  graphdat = {'ngames': [], 'loss': [], 'score': []}
  gamesper1000 = 0
  # rendering
  vis_args = vis_util.setup(env, title='Snake SimpleNet TRAINING')
  # iterations
  episodes = 300_000
  mean_score, mean_loss = 0, 0
  for episode in range(1, episodes+1):
    # get current state
    state = env.getstate()
    # render
    vis_util.render(*vis_args, *state)
    # state vector
    state = state_util.state2vector(*env.getstate())
    # generate agent action
    action = agent.getaction(state, ngames)
    # execute
    reward, done = env.step(action)
    # get next state vector
    nstate = state_util.state2vector(*env.getstate())
    # cache
    agent.cache(state, action, reward, nstate, done)
    # short mem learn
    q, loss = agent.learn(state, action, reward, nstate, done)
    mean_loss += loss
    if done:
      gamesper1000 += 1
      mean_score += env.score
      if env.score > highscore:
        highscore = env.score
        print(f'\n[+] New AI HighScore: {highscore}\n')
      agent.learn_long_mem()
      ngames += 1
      env.reset()
    state = nstate
    if episode % 1000 == 0:
      if gamesper1000 == 0: gamesper1000 = 1
      mean_score /= gamesper1000
      mean_loss /= 1000
      print('----\n'
           f' Episode {episode:05}\n'
           f' Games   {ngames}\n'
           f' Loss    {mean_loss}\n'
           f' Score   {mean_score}\n')
      graphdat['ngames'].append(ngames)
      graphdat['loss'].append(float(mean_loss))
      graphdat['score'].append(mean_score)
      mean_score, mean_loss = 0, 0
      gamesper1000 = 0
  print('[*] TRAINING COMPLETE')
  print(f'[*] AI HighScore during training: {highscore}')
  # save trained model
  name = f'simplenet{highscore}.pth'
  print(f'[*] Saving trained model ({SAVEPATH}/{name})')
  model_util.save(agent.net, SAVEPATH, name)
  # plot mean loss - score
  plot_util.plot(graphdat['ngames'], graphdat['loss'], label='mean loss', color='red')
  plot_util.plot(graphdat['ngames'], graphdat['score'], label='mean score', color='blue')

if __name__ == '__main__':
  train()
