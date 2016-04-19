import logging
import numpy as np
import itertools

from SVM import SVM

class MulticlassSVM(object):
    def __init__(self, kernel, C, K):
        self.kernel = kernel
        self.C = C
        self.K = K

    def fit(self, X_idx, y):
        self.classes = np.unique(y)
        logging.debug('Fitting %s data points with %s different classes '\
                      'with multiclass svm', X_idx.shape[0], len(self.classes))

        self.svms = []
        for class_a, class_b in itertools.combinations(self.classes, 2):
            filtered_X_idx, filtered_y = self.filter_data(X_idx, y, class_a, class_b)

            svm = SVM(self.kernel, self.C, self.K)
            svm.fit(filtered_X_idx, filtered_y)
            self.svms.append((class_a, class_b, svm))

    def predict(self, X_idx):
        logging.debug('Predicting %s data points with multiclass svm', X_idx.shape[0])
        counts = []
        for i in range(len(X_idx)):
            counts.append(dict.fromkeys(self.classes, 0))

        for class_a, class_b, svm in self.svms:
            results = svm.predict(X_idx)
            for idx, result in enumerate(results):
                pred_class = class_a if result == 0 else class_b
                counts[idx][pred_class] += 1             

        result = np.array(
            [max(count.iterkeys(), key=lambda x: count[x]) for count in counts])

        return result

    def score(self, X_idx, y):
        score = np.average(self.predict(X_idx) == y)
        logging.debug('Finished scoring %s data points using multiclass svm with score %s',
                X_idx.shape[0], score)
        return score

    def filter_data(self, X_idx, y, class_a, class_b):
        idx = (y == class_a) | (y == class_b)
        filtered_X_idx = X_idx[idx]
        filtered_y = y[idx]
        filtered_y[filtered_y == class_a] = 0
        filtered_y[filtered_y == class_b] = 1
        return filtered_X_idx, filtered_y
