#!/usr/bin/env python3
import numpy as np
from game.snake import Snake
from game.food import Food
from game.directions import Directions as Dir

# input vec = [distance to danger up, down, left, right, distance to food up, down, left, right]
def state2vector(GRID, snake, food):
  vec = []
  head = snake[0]
  fcoords = food.coords
  d_up, d_down, d_left, d_right = 0, 0, 0, 0
  f_up, f_down, f_left, f_right = 0, 0, 0, 0
  # danger - body
  for part in snake[1:]:
    if [head[0], head[1]-1] == part: d_up = 1
    if [head[0], head[1]+1] == part: d_down = 1
    if [head[0]-1, head[1]] == part: d_left = 1
    if [head[0]+1, head[1]] == part: d_right = 1
  # danger - borders
  if head[1] == 0: d_up = 1
  elif head[1] == GRID[1]-1: d_down = 1
  if head[0] == 0: d_left = 1
  elif head[0] == GRID[0]-1: d_right = 1
  # food
  if head[1] >= fcoords[1]: f_up = 1
  if head[1] < fcoords[1]: f_down = 1
  if head[0] >= fcoords[0]: f_left = 1
  if head[0] < fcoords[0]: f_right = 1
  vec = [d_up, d_down, d_left, d_right, f_up, f_down, f_left, f_right]
  return np.array(vec)

if __name__ == '__main__':
  snake = Snake((0,0), Dir.RIGHT)
  snake.body = [[0,0],
                [0,1],
                [0,2],
                [1,2],
                [2,2],
                [3,2],
                [3,3]]
  food = Food((5,5))
  vec = state2vector((10,10), snake, food)
  print(vec)
