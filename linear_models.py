import numpy as np
import scipy.linalg as LA

class LinearClassifier():
    '''
    Linear model (perceptron) for classification
    '''
    def __init__(self):
        #self.W = np.zeros()
                        
        return
    def fit(self, X, y):
        '''
        Fitting model
        parameters:
        X - matrix of training samples (rowise)
        y - target labels
        returns:
        accuracy on the training set
        '''
        (n,m) = X.shape
        assert(n == y.size)
        X_ = np.hstack((X,np.ones((n,1))))
        self.w = np.zeros((m,1))
        self.b = np.zeros((1,1))
        params = LA.pinv(X_.T.dot(X_)).dot(X_.T.dot(y))
        self.w = params[:m]
        self.b = params[-1]
        accuracy = self.calc_accuracy(X,y)
        return accuracy
    def get_parameters(self):
        return (w,b)
    def predict(self, X):
        h = X.dot(self.w)+ self.b
        h = np.array([1 if x>0 else -1 for x in h])
        return h
    def calc_accuracy(self, X, y_target):
        y = self.predict(X)
        accuracy = np.sum(y == y_target)/np.size(y_target)
        return accuracy 
        