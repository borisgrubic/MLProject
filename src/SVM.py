import logging
import numpy as np
import cvxopt
import cvxopt.solvers

class SVM(object):
    def __init__(self, kernel, C, K=None):
        self.kernel = kernel
        self.C = C
        self.K = K
        cvxopt.solvers.options['show_progress'] = False

    def fit(self, X_idx, y):
        logging.debug('Fitting %s data points with svm(%s, %s)',
                      X_idx.shape[0], self.kernel, self.C)
        y = np.copy(y)
        y[y == 0] = -1

        samples_cnt = X_idx.shape[0]

        # minimise 1/2 a^T * Q * a + p^T * a, w.r.t. a
        # subject to G * a <= h and A * a = b
        Q = cvxopt.matrix(np.outer(y, y) * self.K[np.ix_(X_idx, X_idx)])
        p = cvxopt.matrix(np.ones(samples_cnt) * -1)

        G = cvxopt.matrix(np.vstack((np.identity(samples_cnt) * -1, 
                                     np.identity(samples_cnt))))
        h = cvxopt.matrix(np.vstack((np.zeros((samples_cnt, 1)), 
                                     np.ones((samples_cnt, 1)) * self.C)))

        A = cvxopt.matrix(y, (1, samples_cnt), tc='d')
        b = cvxopt.matrix(0.0)

        result = cvxopt.solvers.qp(Q, p, G, h, A, b)
        a = np.ravel(result['x'])

        # save only data points with Lagrangian multiplier greater than 0
        # because only these points are used in predictions
        idx = a > 1e-7
        self.a = a[idx]
        self.X_idx = X_idx[idx]
        self.y = y[idx]

        # calculate intercept
        cnt = 0
        self.b = 0.0
        for i in range(samples_cnt):
            if a[i] > 1e-7:
                self.b += y[i] - sum(self.a * self.y * self.K[i, idx])
                cnt += 1
        self.b /= float(cnt)

    def predict(self, X_idx):
        logging.debug('Predicting %s data points with svm(%s, %s)',
                      X_idx.shape[0], self.kernel, self.C)
        values = np.array([self.b] * len(X_idx))
        for i in range(len(X_idx)):
            for j in range(self.X_idx.shape[0]):
                values[i] += self.a[j] * self.y[j] * self.K[X_idx[i], self.X_idx[j]]

        return (np.sign(values) + 1) / 2

    def score(self, X_idx, y):
        score = np.average(self.predict(X_idx) == y)
        logging.debug('Finished scoring %s data points with score %s',
                X_idx.shape[0], score)
        return score
