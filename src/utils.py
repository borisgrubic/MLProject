import numpy as np

def gram_matrix(X, kernel):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel(X[i], X[j]) + 1e-5

    return K

