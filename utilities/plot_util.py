#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, label=None, color=None):
  plt.plot(x, y, color=color)
  plt.xlabel('N games')
  plt.ylabel(label)
  plt.show()

if __name__ == '__main__':
  x = [1, 2, 3, 4, 5]
  y = [100, 300, 500, 50, 60]
  plot(x, y, label='test1', color='black')
