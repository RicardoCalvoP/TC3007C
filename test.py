import pandas as pd

df = pd.read_csv("User1_Pre2.csv")
df.head()
X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

m, n = X.shape

# LDL^T implementation

from scipy.linalg import ldl
import numpy as np

P = np.cov(X, rowvar=False)
L, D , _ = ldl(P, lower=True)

D_sqrt = np.zeros_like(D)
for i in range(D.shape[0]):
    D_sqrt[i, i] = np.sqrt(max(D[i, i], 0))

S = L @ D_sqrt
np.set_printoptions(precision=15, suppress=True, linewidth=99999999999999999999999999999999999999)

print(S.shape)

def taylor_series(n):
    dt= 1 / 128 # delta of t
    F = np.eye(n) # Identity Matrix

    for k in range(1, n):
      val = ((dt)**(k))/(k)
      for i in range(n - k):
          j = i + k
          F[i, j] = val
    return F


F = taylor_series(n)

def givens_rotation(F, Qt, St, m):
  assert F.shape == (n, n)
  assert St.shape == (n, n)
  assert Q.shape == (n, n)


  U = np.vstack((F @ St, Qt))

  for j in range(m):
    for i in range(2*m):

      B = np.eye(n)
      a = U[i-1, j]
      b = U[i,   j]

      if b == 0:
        c = 1
        s = 0
      else:
        if abs(b) > abs(a):
          r = a/b
          s = 1/np.sqrt(1 + r**2)
          c = s * r
        else:
          r = b/a
          c = 1/np.sqrt(1 + r**2)
          s = c * r
      row_up = U[i-1, :].copy()
      row_dn = U[i,   :].copy()
      U[i-1, :] = c*row_up - s*row_dn
      U[i,   :] = s*row_up + c*row_dn

  S = U[:m, :].T

  return S

def potter_method(xt, St, yt, H, R):
  x = xt.astype(float).copy()
  S = St.astype(float).copy()
  n = x.size
  I = np.eye(n)

  for i in range(yt.size):
      h = H[i, :]
      r = float(R[i])
      phi = S.T @ h
      alpha = 1.0 / (phi @ phi + r)
      gamma = alpha / (1.0 + np.sqrt(alpha * r))

      S = S @ (I - (alpha * gamma) * np.outer(phi, phi))

      K = S.T @ phi

      innov = yt[i] - h @ x
      x = x + K * innov

  return x, S

#Get the prediction for the
sigma = 0.01      # standard deviation of noise chosen by us
w = np.random.normal(0, sigma, (m, 1))  # vector m x 1

W = np.random.normal(0, sigma, (n, n))  # N x m
Q = np.cov(W, rowvar=False)

for sample in X:
  # New predicated state and
  # proces matrix
  xt = F @ sample + w
  S = givens_rotation(F, Q, S, n)

import matplotlib.pyplot as plt

X = df.select_dtypes(include=[np.number]).to_numpy()

plt.figure(figsize=(24,12))
for i in range(n):
    plt.plot(range(m), X[:, i], label=f"Sensor {i+1}", alpha=0.7)

plt.xlabel("Sample index")
plt.ylabel("Sensor value")
plt.title("Original sensor measurements")
plt.legend(loc="upper right", ncol=3, fontsize=8)
plt.show()
