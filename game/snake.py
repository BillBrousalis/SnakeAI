#!/usr/bin/env python3
from game.directions import Directions as Dir

class Snake:
  COLOR = (0, 110, 9)
  def __init__(self, coords, hdir):
    self.body = []
    if isinstance(coords[0], list): self.body.extend(coords)
    else: self.body.append(coords)
    self.hdir = hdir
    self.tail = coords[-1]

  @property
  def hdir(self):
    return self._hdir

  @hdir.setter
  def hdir(self, d):
    if not isinstance(d, Dir): raise Exception('[!] Error setting hdir, not Dir type')
    self._hdir = d

  @property
  def body(self):
    return self._body

  @body.setter
  def body(self, b):
    if not isinstance(b, list): b = list(b)
    self._body = b

  def __len__(self):
    return len(self.body)

  def __str__(self):
    return '\n'.join([f'({part[0]} , {part[1]})' for part in self.body])

  def __getitem__(self, key):
    if isinstance(key, slice):
      return [self.body[i] for i in range(*key.indices(len(self.body)))]
    if key >= len(self.body) and key < 0:
      raise Exception(f'__getitem__ ERROR: {key} out of range of snake')
    return self.body[key]

  def getcoords(self):
    return self.body[0]

  def move(self):
    self.tail = self.body[-1]
    head = [self.body[0][i]+self.hdir.value[i] for i in range(2)]
    self.body = [head] + self.body[:-1]

  def add(self):
    self.body.append(self.tail)

  def isdead(self):
    return any([self.body[0] == part for part in self.body[1:]])

if __name__ == '__main__':
  x = Snake((0,0), Dir.UP)
