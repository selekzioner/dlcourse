import numpy as np

class KNN:
    
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)
            
        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)
            
    def compute_distances_two_loops(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                dists[i_test, i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
                
        return dists

    def compute_distances_one_loop(self, X):
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.sum(np.abs(X[i_test] - self.train_X), axis = 1)
            
        return dists

    def compute_distances_no_loops(self, X):
        dists = np.sum(np.abs(X[: , None] - self.train_X[None, :]), axis = 2)
        return dists

    def predict_labels_binary(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.bool)
        for i in range(num_test):
            idx = dists[i].argpartition(self.k - 1)
            bincount = np.bincount(self.train_y[idx[:self.k]])
            if bincount.shape[0] > 1:
                pred[i] = bincount[0] < bincount[1]
            else:
                pred[i] = 0
        return pred

    def predict_labels_multiclass(self, dists):
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        class_count = 10
        
        for i in range(num_test):
            idx = dists[i].argpartition(self.k - 1)
            bincount = np.bincount(self.train_y[idx[:self.k]])
            pred[i] = np.where(bincount == np.amax(bincount))[0][0]
            
        return pred
