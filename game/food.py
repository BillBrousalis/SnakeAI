#!/usr/bin/env python3
import random
class Food:
  COLOR = (224, 18, 18)
  def __init__(self, coords):
    self.coords = coords

  @property
  def coords(self):
    return self._coords

  @coords.setter
  def coords(self, x):
    if not isinstance(x, list): x = list(x)
    self._coords = x

  def newcoords(self, snakebody, GRID):
    self.coords = [random.randint(0, g-1) for g in GRID]
    while self.coords in snakebody: 
      self.coords = [random.randint(0, g-1) for g in GRID]
    return self.coords

