import numpy as np
import scipy.linalg as LA
import math

class KNNClassifier():
    '''
    k Nearest Neighbour method realisation for classification problems
    k - model parameter
    '''
    def __init__(self, k=1):
        self.k = k;
    def fit(self, X, y):
        '''
        Fitting model
        parameters:
        X - matrix of training samples (rowise)
        y - target labels
        returns:
        accuracy on the training set
        '''
        assert(X.shape[0]>self.k)
        self._X = X.copy()
        self._y = y.copy()
        accuracy = self.calc_accuracy(X,y)
        return accuracy
        
    def predict(self, X):
        h = np.zeros((X.shape[0],1))  # hypothesis
        N = self._X.shape[0]    # count of samples in memory
        l = X.shape[0] # count of predicted examples        
        for i, x in enumerate(X):
            distances = np.zeros((N,1))        
            distances = [self.euclidian_distance(x, x_mem) for x_mem in self._X]
            #print(distances)
            #distances_sorted = np.sort(distances)
            sorted_indices = np.argsort(distances)
            #kNN_set = self._X[sorted_indices][:self.k]
            kNN_y = self._y[sorted_indices][:self.k]
            pred_avg = np.sum(kNN_y)/self.k
            h[i] = pred_avg
            #print(kNN_set)
        h = np.array([1 if x>0 else -1 for x in h])      
        return h
            
    def calc_accuracy(self, X, y_target):
        y = self.predict(X)
        accuracy = np.sum(y == y_target)/np.size(y_target)
        return accuracy 
    def euclidian_distance(self, x1, x2):
        assert(x1.size == x2.size)
        #assert(x1.shape[0] == 1 or x1.shape[1] == 1)
        dist = math.sqrt(sum([(x1[i] - x2[i])**2 for i in range(x1.size)]))    
        return dist
        
    