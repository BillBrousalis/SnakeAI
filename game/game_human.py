#!/usr/bin/env python3
import random
import pygame as pg
from game.snake import Snake
from game.food import Food
from game.keys import Keys
from game.directions import Directions as Dir

def grid2res(gcoords, RES, GRID):
  assert all([(gcoords[i] >= 0 and gcoords[i] < GRID[i]) for i in range(len(gcoords))])
  return tuple([int(gcoords[i]*RES[i]/GRID[i]) for i in range(len(gcoords))])

def render(screen, snake, food, BG, RES, GRID):
  screen.fill(BG)
  # food render
  fcoords = grid2res(food.coords, RES, GRID)
  pg.draw.rect(screen, food.COLOR, (fcoords[0], fcoords[1], RES[0]/GRID[0]-2, RES[1]/GRID[1]-2))
  # snake render
  for scoords in snake.body:
    scoords = grid2res(scoords, RES, GRID)
    pg.draw.rect(screen, snake.COLOR, (scoords[0], scoords[1], RES[0]/GRID[0]-2, RES[1]/GRID[1]-2))
  # update entire screen
  pg.display.update()

def main():
  # const
  BG = (15, 15, 15)
  GRID = (12, 12)
  RES = (GRID[0]*30, GRID[1]*30)
  FPS = 5
  assert all([RES[i]%GRID[i] == 0 for i in range(2)])
  # setup
  pg.init()
  screen = pg.display.set_mode(RES)
  pg.display.set_caption('Vim Snake')
  pg.mouse.set_visible(False)
  # background surface
  bg = pg.Surface(screen.get_size()).convert()
  bg.fill(BG)
  # screen
  screen.blit(bg, (0, 0))
  pg.display.flip()
  # snake instance - coordinates - direction
  snake = Snake((0, 0), Dir.RIGHT)
  # food instance - coordinates
  food_coords = (random.randint(0, GRID[0]-1), random.randint(0, GRID[1]-1))
  food = Food(food_coords)
  # scorekeeping
  score = 0
  # gameloop
  clock = pg.time.Clock()
  isRunning = True
  while isRunning:
    # refresh rate
    clock.tick(FPS)
    # keypresses
    for event in pg.event.get():
      if event.type == pg.QUIT:
        isRunning = False
      elif event.type == pg.KEYDOWN:
        key = Keys(event.key)
        # directional check - only helps with snake of size 2 (worth processing it?)
        #if key.value == -snake.hdir.value:
          #print('DEAD')
          #exit()
        if key == Keys.UP: snake.hdir = Dir.UP
        elif key == Keys.DOWN: snake.hdir = Dir.DOWN
        elif key == Keys.RIGHT: snake.hdir = Dir.RIGHT
        elif key == Keys.LEFT: snake.hdir = Dir.LEFT
    # apply move
    snake.move()
    # coordinates checks
    snakecoords = snake.getcoords()
    if snake.isdead() or any([(snakecoords[i] < 0 or snakecoords[i] >= GRID[i]) for i in range(2)]):
      print("DEAD")
      exit()
    elif snake.getcoords() == food.coords:
      snake.add()
      food.newcoords(snake.body, GRID)
      score += 1
      print(f'----\nSCORE: {score}')
    # rendering
    render(screen, snake, food, BG, RES, GRID)

if __name__ == '__main__':
  main()
