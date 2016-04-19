import sys
import os
import random
import numpy as np
import logging
from sklearn.cross_validation import StratifiedKFold

from multiclass_SVM import MulticlassSVM
import kernels
import utils

tmp_folder = '../tmp'

class Analyzer(object):
    def __init__(self, data_file, C_steps=17, G_steps=8, 
                 gram_folder='../data/gram_matrices',
                 tmp_folder='../tmp'):
        self.data_file = data_file
        self.gram_folder = gram_folder

        self.C_steps = C_steps
        self.gamma_steps = G_steps
        self.tmp_folder = tmp_folder

    def read_data(self, feature_subset=None):
        logging.debug('Reading data')
        X = []
        y = []
        dataPhaseState = 0
        for line in open(self.data_file):
            if len(line) == 0:
                continue
            if line.startswith('!'):
                if line.startswith('!Sample_characteristics_ch1	"Illness'):
                    y = [str[:str.find('"')] for str in line.split('Illness: ')[1:]]
                elif line.startswith('!series_matrix_table_begin'):
                    dataPhaseState = 1
                elif line.startswith('!series_matrix_table_end'):
                    dataPhaseState = 0
            elif dataPhaseState > 0:
                if dataPhaseState == 1:
                    dataPhaseState = 2
                else:
                    data = line.split()[1:]
                    X.append([float(x) for x in data])

        self.y = np.array(y)
        self.X = np.transpose(X)
        if feature_subset is not None:
            self.X = self.X[:,feature_subset]

    def preprocess_data(self):
        logging.debug('Preprocessing data')
        for i in range(self.X.shape[1]):
            stdev = np.std(self.X[:,i])
            mean = np.mean(self.X[:,i])
            self.X[:,i] = (self.X[:,i] - mean) / stdev

        y = self.y
        self.classes = np.unique(y)
        self.y = np.array([0] * len(y))
        for idx, class_name in enumerate(self.classes):
            self.y[y == class_name] = idx
            print class_name, sum(y == class_name)

    def analyze(self):
        logging.debug('Analysing data')
        self.read_data()
        self.preprocess_data()
        models = self.get_models()

        kfold = StratifiedKFold(self.y, 5)
        result = 0.0
        for train_idx, test_idx in kfold:
            svm, _ = self.select_model(train_idx, self.y[train_idx], models)
            svm.fit(train_idx, self.y[train_idx])

            score = svm.score(test_idx, self.y[test_idx])
            logging.info('Best model achieved score %s on test data', score)

            result += score
        result /= 5.0

        logging.info('Test score: %s', result)
        print 'Test score: %s' % result
        return result

    def select_model(self, train_idx, train_y, models):
        logging.debug('Selecting model')

        best_model = None
        best_score = 0.0
        kfold = StratifiedKFold(train_y, 5)
        for model in models:
            result = 0.0
            for train_part, validate_part in kfold:
                model.fit(train_idx[train_part], train_y[train_part])
                result += model.score(train_idx[validate_part], train_y[validate_part])
            result /= 5.0

            logging.info('Model: (%s, %s) achieved score %s',
                    model.kernel, model.C, result)

            if result > best_score:
                best_score = result
                best_model = model

        logging.info('Best model: (%s, %s) achieved score %s',
                best_model.kernel, best_model.C, best_score)
        return best_model, best_score

    def get_models(self):
        logging.debug('Generating models')
        models = []
        models.extend(self.get_linear_models())
        models.extend(self.get_gaussian_models())
        return models

    def get_linear_models(self):
        logging.debug('Generating linear models')
        kernel = kernels.kernel_wrapper(kernels.linear)
        K = self.get_gram_matrix('linear_gram', kernel)
        models = []
        for C in range(self.C_steps):
            models.append(MulticlassSVM(kernel, 0.001 * 2 ** C, K))
        return models

    def get_gaussian_models(self):
        logging.debug('Generating gaussian models')
        models = []
        for G in range(self.gamma_steps):
            logging.debug('Generating gaussian models with gamma=%s', 0.0001 * 10 ** G)
            kernel = kernels.kernel_wrapper(kernels.gaussian, gamma=0.0001 * 10 ** G)
            K = self.get_gram_matrix('gaussian_gram_%s' % G, kernel)
            for C in range(self.C_steps):
                models.append(MulticlassSVM(kernel, 0.001 * 2 ** C, K))
        return models

    def get_gram_matrix(self, file_base, kernel):
        file_path = None
        if self.gram_folder is not None:
            file_path = '%s/%s' % (self.gram_folder, file_base)

        if file_path is not None and os.path.exists(file_path):
            K = self.read_gram_matrix(file_path)
        else:
            K = utils.gram_matrix(self.X, kernel)
            self.write_gram_matrix(K, '%s/%s' % (self.tmp_folder, file_base))
        return K

    def write_gram_matrix(self, K, file_path):
        with open(file_path, 'w') as fout:
            n = K.shape[0]
            for i in range(n):
                for j in range(n):
                    fout.write('%s ' % K[i, j])
                fout.write('\n')

    def read_gram_matrix(self, file_path):
        K = []
        for line in open(file_path):
            K.append([float(x) for x in line.split()])
        return np.array(K)

    def train_test_split(self):
        logging.debug('Splitting data')
        train_indices = np.array([True] * self.X.shape[0])
        r = np.arange(self.X.shape[0])
        for class_num in range(len(self.classes)):
            idx = self.y == class_num
            cnt = sum(idx)
            test_indices = np.array(
                    random.sample(r[idx], int(cnt * 0.4)))
            train_indices[test_indices] = False
 
        return train_indices

    def get_feature_scores(self):
        logging.debug('Getting feature scores')
        self.read_data()
        self.preprocess_data()
        feature_score = []
        train_indices = self.train_test_split()
        train_X = self.X[train_indices]
        train_y = self.y[train_indices]
        test_X = self.X[~train_indices]
        test_y = self.y[~train_indices]
        for i in range(self.X.shape[1]):
            K = utils.gram_matrix(train_X[:,i].reshape(train_X.shape[0], 1), kernels.gaussian)
            svm = MulticlassSVM(kernels.gaussian, 1.0, K)
            model, score = self.select_model(np.arange(train_X.shape[0]), train_y, [svm])
            feature_score.append((score, i))

        feature_score.sort()
        feature_score.reverse()

        logging.info('Sorted feature scores: %s' % feature_score)
        with open('%s/feature_score' % self.tmp_folder, 'w') as fout:
            fout.write('%s\n' % train_indices)
            for score in feature_score:
                fout.write('%s: %s\n' % (score[1], score[0]))

    def feature_selection(self):
        logging.debug('Getting feature scores')
        kernel_linear = kernels.kernel_wrapper(kernels.linear)
        kernel_gaussian = kernels.kernel_wrapper(kernels.gaussian, gamma=0.01)
        train_indices = []
        best_features = []
        for line in open('../results/feature_selection/feature_score'):
            if ':' in line:
                best_features.append(int(line.split(':')[0]))
            else:
                line = line.strip()
                if line.startswith('['):
                    line = line[1:]
                elif line.endswith(']'):
                    line = line[:-1]
                for s in line.split():
                    if s == 'True':
                        train_indices.append(True)
                    else:
                        train_indices.append(False)
        train_indices = np.array(train_indices)
        best_features = np.array(best_features)

        with open('%s/best_features' % self.tmp_folder, 'w') as fout:
            for subset_size in range(10, min(best_features.shape[0], 1001), 10):
                logging.debug("Doing %s", subset_size)
                self.read_data(best_features[:subset_size])
                self.preprocess_data()
                K_linear = utils.gram_matrix(self.X, kernel_linear)
                K_gaussian = utils.gram_matrix(self.X, kernel_gaussian)
                models = []
                for C in range(self.C_steps):
                    models.append(MulticlassSVM(kernel_linear, 0.001 * 2 ** C, K_linear))
                models.append(MulticlassSVM(kernel_gaussian, 8.192, K_gaussian))
                models = []
                K_linear = self.get_gram_matrix('linear_gram', kernel_linear)
                for C in range(self.C_steps):
                    models.append(MulticlassSVM(kernel_linear, 0.001 * 2 ** C, K_linear))

                train_idx = np.arange(self.X.shape[0])[train_indices]
                test_idx = np.arange(self.X.shape[0])[~train_indices]
                best_score = 0.0
                best_model = None
                for model in models:
                    model.fit(train_idx, self.y[train_idx])
                    score = model.score(test_idx, self.y[test_idx])
                    logging.debug('%s: (%s %s) - %s', subset_size, model.kernel, model.C, score)

                    if score > best_score:
                        best_score = score
                        best_model = model

                logging.info('%s: (%s, %s) - %s', subset_size, best_model.kernel, best_model.C, best_score)
                fout.write('%s: %s\n' % (subset_size, best_score))

if __name__ == '__main__':
    logging.basicConfig(filename='%s/data_analysis.log' % tmp_folder, 
                        filemode='w', 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    data_file = sys.argv[1]

    analyzer = Analyzer(data_file)
    # analyzer.analyze()
    # analyzer.get_feature_scores()
    analyzer.feature_selection()
