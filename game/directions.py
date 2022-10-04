#!/usr/bin/env python3
import enum
class Directions(enum.Enum):
  UP = (0, -1)
  DOWN = (0, 1)
  RIGHT = (1, 0)
  LEFT = (-1, 0)
