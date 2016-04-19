import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.svm import SVC
from SVM import SVM
import kernels
import utils
import logging

names = ["sklearn - Linear SVM", "sklearn - RBF SVM", 
         "SVM Linear", "SVM Gaussian"]
classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    SVM(kernels.kernel_wrapper(kernels.linear), 0.025),
    SVM(kernels.kernel_wrapper(kernels.gaussian), 1.0)]

tmp_folder = '../tmp'

def compare_classifiers(names, classifiers):
    scores = {name: [0, 0, 0] for name in names}
    for test in range(100):
        logging.debug('Running test %s', test)
        X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                                   random_state=1, n_clusters_per_class=1)
        rng = np.random.RandomState(2)
        X += 2 * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        datasets = [make_moons(noise=0.3, random_state=0),
                    make_circles(noise=0.2, factor=0.5, random_state=1),
                    linearly_separable
                    ]

        for i, ds in enumerate(datasets):
            X, y = ds
            X = StandardScaler().fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

            x_values = ['%s:%s' % (x[0], x[1])  for x in X_train]
            idx = np.array([False] * X.shape[0])
            indices = np.arange(X.shape[0])
            for j, x in enumerate(X):
                if '%s:%s' % (x[0], x[1]) in x_values:
                    idx[j] = True
            K_linear = utils.gram_matrix(X, kernels.linear)
            K_gaussian = utils.gram_matrix(X, kernels.gaussian)
            classifiers[2].K = K_linear
            classifiers[3].K = K_gaussian

            for name, clf in zip(names, classifiers):
                if name.startswith("sklearn"):
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)
                else:
                    clf.fit(indices[idx], y[idx])
                    score = clf.score(indices[~idx], y[~idx])

                scores[name][i] += score / 100.0
    return scores

if __name__ == "__main__":
    logging.basicConfig(filename='%s/comparison.log' % tmp_folder, 
                        filemode='w', 
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    scores = compare_classifiers(names, classifiers)
    for name in scores:
        logging.info('Result for classifier %s: %s', name, scores[name])
