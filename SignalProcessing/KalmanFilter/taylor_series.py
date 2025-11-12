import numpy as np
from math import factorial

def taylor_series_matrix(A, delta_t, n_terms):

    n = A.shape[0]

    F = np.identity(n)

    A_dt = A * delta_t

    for k in range(1, n_terms + 1):

        potencia_A_dt = np.linalg.matrix_power(A_dt, k)

        factorial_k = factorial(k)

        termino = potencia_A_dt / factorial_k

        F = F + termino

    return F