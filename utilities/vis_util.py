#!/usr/bin/env python3
import pygame as pg
from game.directions import Directions as Dir

def _assert(gcoords, RES, GRID):
  if all([(gcoords[i] >= 0 and gcoords[i] < GRID[i]) for i in range(len(gcoords))]): return
  print(gcoords)
  print(RES)
  print(GRID)
  print('[-] grid2res -> Assertion Error')

def grid2res(gcoords, RES, GRID):
  _assert(gcoords, RES, GRID)
  return tuple([int(gcoords[i]*RES[i]/GRID[i]) for i in range(len(gcoords))])

def render(screen, BG, RES, GRID, snake, food):
  screen.fill(BG)
  # food render
  fcoords = grid2res(food.coords, RES, GRID)
  pg.draw.rect(screen, food.COLOR, (fcoords[0], fcoords[1], RES[0]//GRID[0]-2, RES[1]//GRID[1]-2))
  # snake render
  for scoords in snake.body:
    scoords = grid2res(scoords, RES, GRID)
    pg.draw.rect(screen, snake.COLOR, (scoords[0], scoords[1], RES[0]//GRID[0]-2, RES[1]//GRID[1]-2))
  # update entire screen
  pg.display.update()

def setup(env, title='SnakeAI'):
  BG = (10, 10, 10)
  GRID = env.GRID
  RES = (GRID[0]*35, GRID[1]*35)
  assert all([RES[i]%GRID[i] == 0 for i in range(2)])
  # setup
  pg.init()
  screen = pg.display.set_mode(RES)
  pg.display.set_caption(title)
  pg.mouse.set_visible(False)
  # background surface
  bg = pg.Surface(screen.get_size()).convert()
  bg.fill(BG)
  # screen
  screen.blit(bg, (0, 0))
  pg.display.flip()
  # use as *args with render(*args, GRID, snake, food)
  return (screen, BG, RES)
