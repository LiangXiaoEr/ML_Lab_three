import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.w_clf = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.w = 0
        self.alphas = []
        self.w_clfs = []
        self.error = []
        self.accuracy = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        self.w = np.ones(X.shape[0])/X.shape[0]
        for i in range(self.n_weakers_limit):
            clf = self.w_clf.fit(X,y,sample_weight=self.w)
            y_pred = clf.predict(X)
            err = np.sum((y_pred != y.reshape(-1,))*self.w)
            acc = np.mean(y_pred == y.reshape(-1,))
            if err > 0.5:
                break
            alpha = 1/2 * np.log((1-err)/err)
            self.w *= np.exp(-y.reshape(-1,)*alpha*y_pred)
            self.w /= np.sum(self.w)
            self.alphas.append(alpha)
            self.w_clfs.append(clf)
            self.error.append(err)
            self.accuracy.append(acc)


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        scores = np.zeros(X.shape[0])
        for i in range(len(self.w_clfs)):
            scores += self.alphas[i] * self.w_clfs[i].predict(X)
        return scores

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        y_pred = self.predict_scores(X)
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = -1
        return y_pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)