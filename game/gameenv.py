#!/usr/bin/env python3
import random
import pygame as pg
from game.snake import Snake
from game.food import Food
from game.keys import Keys
from game.directions import Directions as Dir

class GameEnv:
  def __init__(self, mode=None):
    if mode == 'train': self.GRID = (8, 8)
    elif mode == 'test': self.GRID = (20, 20)
    else: raise Exception(f'Unknown GameEnv mode: {mode}\nOptions: "train" / "test"')
    self.isrunning = True
    self.reset()
  
  def reset(self):
    self.isrunning = True
    # snake instance - random coordinates - direction
    rcoords = [random.randint(2, self.GRID[0]-1), random.randint(2, self.GRID[1]-1)]
    # start with 3-block body 
    rcoords = [rcoords, [rcoords[0]-1, rcoords[1]], [rcoords[0]-2, rcoords[1]]]
    self.snake = Snake(rcoords, Dir.RIGHT)
    # food instance - coordinates
    food_coords = [random.randint(0, self.GRID[0]-1), random.randint(0, self.GRID[1]-1)]
    self.food = Food(food_coords)
    # scorekeeping
    self.score = 0
    self.iter = 0

  def step(self, action):
    self.iter += 1
    if self.iter > self.GRID[0]*20:
      self.isrunning = False
      reward = -10
    if self.isrunning:
      reward = 0.0
      # action to Dir
      if action[0] == 1: action = Dir.UP
      elif action[1] == 1: action = Dir.DOWN
      elif action[2] == 1: action = Dir.LEFT
      elif action[3] == 1: action = Dir.RIGHT
      self.snake.hdir = action
      # apply move
      self.snake.move()
      # coordinates checks
      snakecoords = self.snake.getcoords()
      if self.snake.isdead() or any([(snakecoords[i] < 0 or snakecoords[i] >= self.GRID[i]) for i in range(2)]):
        self.isrunning = False
        reward = -10
      elif self.snake.getcoords() == self.food.coords:
        self.snake.add()
        self.food.newcoords(self.snake.body, self.GRID)
        self.score += 1
        self.iter = 0
        reward = 10
    return reward, not self.isrunning

  def getstate(self):
    return (self.GRID, self.snake, self.food)
