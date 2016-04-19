import random
from unittest import TestCase

import numpy as np
import kernels
from SVM import SVM

class TestIntegrationSVM(TestCase):
    def test_simple(self):
        X = np.array([[1.0, 1.0],[2.0, 2.0],[3.0, 3.0],[4.0, 4.0],
                      [1.0, 1.0],[2.4, 2.4],[2.6, 2.6],[4.0, 4.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        K = self.gram_matrix(X, kernels.linear)
        svm = SVM(kernels.linear, 1.0, K)
        svm.fit(np.arange(4), y)

        result = svm.predict(np.arange(4,8))
        np.testing.assert_allclose(result, [0, 0, 1, 1])

    def run_test(self, X, y, kernel):
        n = int(X.shape[0] * 0.8)
        K = self.gram_matrix(X, kernel)

        svm = SVM(kernel, 1.0, K)
        svm.fit(np.arange(n), y[:n])
        score = svm.score(np.arange(n, X.shape[0]), y[n:])

        return score

    def test_linear_kernel_linearly_separable(self):
        X, y = self.generate_linearly_separable()
        score = self.run_test(X, y, kernel=kernels.linear)
        self.assertTrue(score > 0.9)

    def test_gaussian_kernel_linearly_separable(self):
        X, y = self.generate_linearly_separable()
        score = self.run_test(X, y, kernel=kernels.gaussian)
        self.assertTrue(score > 0.9)

    def test_gaussian_kernel_linearly_nonseparable(self):
        X, y = self.generate_linearly_nonseparable()
        score = self.run_test(X, y, kernel=kernels.gaussian)
        self.assertTrue(score > 0.9)

    def generate_linearly_separable(self):
        points_num = 500
        k = random.random()
        n = 0.5 - random.random()
        X = []
        y = []
        while len(X) < points_num:
            newX = random.random()
            newY = random.random()
            value = newY - k * newX + n
            if abs(value) > 1e-4:
                X.append([newX, newY])
                y.append(1 if value > 0 else 0)
        return np.array(X), np.array(y)

    def generate_linearly_nonseparable(self):
        points_num = 500
        X = []
        y = []
        while len(X) < points_num:
            newX = random.random()
            newY = random.random()
            if abs(newX - 0.25) > 1e-3 and abs(newY - 0.75) > 1e-3:
                X.append([newX, newY])
                if newX < 0.25 or newX > 0.75:
                    y.append(0)
                else:
                    y.append(1)
        return np.array(X), np.array(y)

    def gram_matrix(self, X, kernel):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = kernel(X[i], X[j]) + 1e-5

        return K
