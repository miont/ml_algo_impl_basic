# -*- coding: utf-8 -*-
"""
Fully connected multilayer perceptron
"""
import numpy as np
import scipy as scp
import scipy.linalg as LA
import math
from enum import Enum

class ActivationFunc(Enum):
    SIGMOID = 1
    HYPERBOLIC_TANGENT = 2
    RLU = 3
    SOFTMAX = 4
    def get(func_type):
        func = None
        der_func = None
        if func_type == ActivationFunc.SIGMOID :
            func = ActivationFunc.sigmoid
            der_func = ActivationFunc.der_sigmoid
        elif func_type == ActivationFunc.SOFTMAX:
            func = ActivationFunc.softmax
            der_func = ActivationFunc.der_softmax
        elif func_type == ActivationFunc.HYPERBOLIC_TANGENT:
            func = ActivationFunc.htan
            der_func = ActivationFunc.der_htan
        elif func_type == ActivationFunc.RLU :
            func = ActivationFunc.rlu
            der_func = ActivationFunc.der_rlu
        return func, der_func
        
    def sigmoid(x):
        return np.ones((x.size))/(np.ones((x.size)) + + np.exp(-x))
    def der_sigmoid(x):
        return ActivationFunc.sigmoid(x)*(1-ActivationFunc.sigmoid(x))
    
    def htan(x):
        pass
    def der_htan(x):
        pass
    
    def rlu(x):
        res = x.copy()
        for i in range(x.size):
            res[i] = 0
            if x[i]>0:
                res[i] = x[i]
        return res
        
    def der_rlu(x):
        res = x.copy()
        for i in range(x.size):
            res[i] = 0
            if x[i]>0:
                res[i] = 1
        return res
        
    def softmax(x):
        return np.exp(x)/np.sum(np.exp(x))
    def der_softmax(x):
        return ActivationFunc.softmax(x)*(1-ActivationFunc.softmax(x))  
    def identity(x):
        return x
    def der_identity(x):
        return np.ones(x.size)
            
class FeedforwardNeuralNet():
    def __init__(self, layers_size=(2,1), activation=ActivationFunc.SIGMOID, output_func=ActivationFunc.SIGMOID):
        if len(layers_size)<2:
            raise Exception('Number of layers must be atleast 2.')
        for x in layers_size:
            assert(x>0)
        self.num_layers = len(layers_size)
        self.num_hiden_layers = self.num_layers-2
        self.weights = []
        for i in range(self.num_layers-1):
            self.weights.append(np.zeros((layers_size[i+1], layers_size[i]+1)))
        self.layers_size = layers_size
        self.activation, self.der_activ = ActivationFunc.get(activation)
        self.output_func, self.der_output = ActivationFunc.get(output_func)
        self.class_labels = np.array((-1, 1))
        assert(self.activation != None)
        
        pass
    
    def fit(self, samples, target, num_epoch = 10, eta = 0.1):
        (n, m) = samples.shape
#        print(target.shape[0], n, self.layers_size)
#        assert((len(target.shape) == 1 and self.layers_size[-1] == 1) or (target.shape[1] == self.layers_size[-1]))
        target_classes = target.copy()
        target, self.class_labels = self.encode_classes(target)     
        self.initialize_weights()        
        finish = False
        epoch = 0
        while(not finish):
            epoch += 1
            print('-------------- epoch %d -------------' %(epoch))
            indexes = np.random.permutation(list(range(n))).tolist()
            max_weights_change = 0
            mean_squared_error = 0
            mean_rel_error = 0
            max_rel_error = 0
            misclassified_count = 0
            for i in range(n):
                sample_idx = indexes[0]
                x = samples[sample_idx]
                del indexes[0]
                output, weighted_sums, signals = self.forward_pass(x)
                local_gradients = self.backprop(signals, weighted_sums, target[sample_idx])
                weights_change = self.calc_weights_correction(signals, local_gradients, eta)
                weights_change_sample = self.update_weights(weights_change)
                if(weights_change_sample > max_weights_change):
                    max_weights_change = weights_change_sample
#                predicted_label = 0              
#                if(output>0.5):
#                    predicted_label = 1
                predicted_label = self.class_labels[output.argmax()]
                if(predicted_label != target_classes[sample_idx]):
                    misclassified_count += 1
                mean_squared_error += LA.norm((target[sample_idx] - output))**2
#                rel_error = LA.norm(target[sample_idx] - output)/LA.norm(target[sample_idx])
#                if rel_error > max_rel_error:
#                    max_rel_error = rel_error
#                mean_rel_error += rel_error
#            self.calc_epoch_stat()
            mean_squared_error /= n
#            mean_rel_error /= n
            print('MAX WEIGHTS RELATIVE CHANGE = %.3f%%' % (max_weights_change*100))
            print('MSE = %.3f' % (mean_squared_error))
            print('CLASSIFICATION ERROR = %.2f%%' % (misclassified_count/n*100))
#            print('MEAN RELATIVE ERROR = %.3f%%' % (mean_rel_error*100))
#            print('MAX RELATIVE ERROR = %.3f%%' % (max_rel_error*100))
            if (epoch >= num_epoch):
                finish = True
            
        
    def initialize_weights(self):
        '''
        Random weight initialization according to (LeCun, 1993)
        '''
        for layer in range(self.num_layers-1):           
            self.weights[layer] = np.random.normal(scale=1.0/math.sqrt(self.layers_size[layer]+1), size=(self.layers_size[layer+1], self.layers_size[layer]+1))
        return
    def forward_pass(self, x):
#        print('-----> forward pass')
        weighted_sums = [];        
        signals = [];
        weighted_sums.append(x)    
        signals.append(x)
        y = x.copy()
        for i in range(self.num_layers-1):          
            W = self.weights[i]
            y = np.hstack((1,y))    
            v = W.dot(y)
            weighted_sums.append(v)
            if i<self.num_layers-1:
                y = self.activation(v)
            else:
                y = self.output_func(v)
            signals.append(y)
                  
        output = signals[-1]
#        print(output, weighted_sums[1:], signals[1:])
        return output, weighted_sums, signals
        
    def backprop(self, signals, induced_local_fields, target):
#        print('-----> backpropagation')
        local_gradients = [0]*self.num_layers
        output = signals[-1]
        error = target - output
        local_gradients[self.num_layers-1] = error*self.der_output(induced_local_fields[-1])
        for layer in reversed(range(0, self.num_layers-1)):
            W = self.weights[layer]
            v = induced_local_fields[layer]
            y  = signals[layer]
            y = np.hstack((1, y))
            
#            print(self.der_activ(v).shape, W.shape, local_gradients[layer+1].shape)
            delta = self.der_activ(v)*W[:, 1:].T.dot(local_gradients[layer+1])
            local_gradients[layer] = delta 
#        print(local_gradients)
        return local_gradients
        
    def calc_weights_correction(self, signals, local_gradients, eta):
        weights_change = []        
        for layer in range(self.num_layers-1):
            delta = local_gradients[layer+1]
            y = signals[layer]
            y = np.hstack((1, y))
            y.shape = (y.size,1)
            delta.shape = (delta.size,1)
            dW = eta*delta.dot(y.T)
            weights_change.append(dW)
        return weights_change
        
    def update_weights(self, weights_change):
        max_update = 0
        small_val = 1e-10   # to prevent division by zero
        for layer in range(self.num_layers-1):
            W = self.weights[layer]
            dW = weights_change[layer]
            W_upd = W +dW
            self.weights[layer] = W_upd
            rel_update = np.max(np.abs(dW/(W + small_val)))
            if rel_update > max_update:
                max_update = rel_update
        return max_update
    def calc_epoch_stat(self):
        pass
    
    def predict(self, X):
        n = X.shape[0]
        output_vect = np.zeros((n,self.layers_size[-1]))
        for i in range(n):
            x = X[i]
            output,_,_ = self.forward_pass(x)
            output_vect[i] = output
        if self.output_func == ActivationFunc.sigmoid:
            return np.array([1 if y>0.5 else -1 for y in output_vect])
        elif self.output_func == ActivationFunc.softmax:
            return self.decode_classes(output_vect, self.class_labels)
        return output_vect
        
    def calc_accuracy(self, X, y_target):
        y = self.predict(X)
        accuracy = np.sum(y == y_target)/np.size(y_target)
        return accuracy 
    
    def encode_classes(self, y):
        n = y.size
        class_labels = np.unique(y)
        num_classes = class_labels.size 
        y_dummy = np.zeros((n,num_classes))
        for i in range(n):
            y_dummy[i, class_labels.tolist().index(y[i])] = 1
        return y_dummy, class_labels
    def decode_classes(self, y, class_labels):
        assert(y.shape[1] == class_labels.size)        
        n = y.shape[0]
        g = np.array([class_labels[y[i].argmax()] for i in range(n)])
        return g
    
        
if __name__ == '__main__':
    net = FeedforwardNeuralNet(layers_size=(2,1))
    

    
    