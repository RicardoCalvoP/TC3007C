import numpy as np
import scipy as sp
from givens import givens
from ldl import ldl
from kalman import kalman

from taylor_series import taylor_series

def read_csv(filename):
  data = np.genfromtxt(filename, delimiter=',', skip_header=1)
  return np.array(data)

def main():
  data = read_csv('User1_Pre2.csv')
  F = taylor_series(data, data.shape[1])
  print(F)
  for line in data:
    #print(line)
    pass

if __name__ == "__main__":
  main()