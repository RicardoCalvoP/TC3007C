
import random
import matplotlib.pyplot as plt

def plot(Q1, Q2, iterations, alpha):
  plt.plot(Q1, label='Arm 1 (P=0.4, W=10)')
  plt.plot(Q2, label='Arm 2 (P=0.03, W=100)')
  plt.xlabel('Iterations')
  plt.ylabel('Estimated Value Q')
  plt.title(f'Estimated Values over {iterations} Iterations (alpha={alpha})')
  plt.legend()
  plt.show()

def reward(prob, w):
  if random.random() < prob:
    return w
  else:
    return 0

def multi_armed_bandit(Q_k_1, alpha, p, w):
  """
  Q: Estimated value of the arm
  alpha: Learning rate
  p: Probability of winning
  w: Reward for winning
  """
  r = reward(p, w)
  Q_new = Q_k_1 + alpha * (r - Q_k_1)
  return Q_new

if __name__ == "__main__":
  """
  Our list P is the probabilitiy of winning, each index corresponds to an arm.
  The list W is the reward for winning, each index corresponds to an arm.
  """

  # Set the proboability of each hand
  P = [0.4, 0.03]
  # Set the reward of each hand
  W = [10, 100]
  # Set the number of iterations
  iterations = [10, 100, 1000]
  # Set the learning rate
  alpha = [ 0.1, 0.01, 0.001]

  """
  In this example we will iterate over different learning reates through different
  number of iterations. For each combination we will plot the estimated value of
  each arm over time.
  """
  for a in alpha:
    for iteration in iterations:
      Q1 = [0]
      Q2 = [0]
      for k in range(1,iteration + 1):
        Q1_k = multi_armed_bandit(Q1[k-1], a, P[0], W[0])
        Q2_k = multi_armed_bandit(Q2[k-1], a, P[1], W[1])

        Q1.append(Q1_k)
        Q2.append(Q2_k)
      plot(Q1, Q2, iteration, a)

  """
  In this example we will calculate the learning rate as 1/k where k is the
  current iteration. We will plot the estimated value of each arm over time.
  """
  for iteration in iterations:
      Q1 = [0]
      Q2 = [0]
      for k in range(1,iteration + 1):
        Q1_k = multi_armed_bandit(Q1[k-1], 1/k, P[0], W[0])
        Q2_k = multi_armed_bandit(Q2[k-1], 1/k, P[1], W[1])

        Q1.append(Q1_k)
        Q2.append(Q2_k)
      plot(Q1, Q2, iteration, f'1/k')