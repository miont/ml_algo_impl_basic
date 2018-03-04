# -*- coding: utf-8 -*-
"""
Zip codes multiclass classification
"""
import numpy as np
import sys
from linear_models import LinearClassifier
from knn_models import KNNClassifier
from feedforward_NN import FeedforwardNeuralNet, ActivationFunc
import time
import datetime
import matplotlib.pyplot as plt

def map_labels(y, labels, reverse=False):
    '''
    Mapping labels to [-1, 1]
    '''
    if not reverse:
        return np.array([1 if x==labels[0] else -1 for x in y])
    else:
        return np.array([labels[0] if x==1 else labels[1] for x in y])

def read_data(file_name):
    '''
    Чтение данных из файла
    '''
    try:
        with open(file_name, 'rt') as f:
            data_list = [line.rstrip().split(' ') for line in f]
        data = np.array(data_list).astype(np.float)
    except IOError:
        print('Cannot open file ', file_name)
    except Exception as exc:
        print('Unexpected error: ', sys.exc_info()[0])
        print(exc)
        raise exc
    print('Data is loaded: %d lines, %d columns ' % (data.shape[0], data.shape[1]))
    return data

if __name__ == '__main__':
    
    print('Loadind train data')      
    data_train = read_data('data/zip.train')
    print('Loadind test data')      
    data_test = read_data('data/zip.test')
    
    X_train = data_train[:,1:]
    y_train = data_train[:,0]    
    print('%d examples in train set' % (X_train.shape[0]))
    X_test = data_test[:,1:]
    y_test = data_test[:,0]
    print('%d examples in test set' % (X_test.shape[0]))
    num_train_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    class_labels = np.unique(y_train)
    num_classes = class_labels.size
         
    print('---Feedforward neural net---')
    net_config = (num_features,100,num_classes)
#        net_config = (num_features,1)
    neural_model = FeedforwardNeuralNet(layers_size=net_config, activation=ActivationFunc.SIGMOID, output_func=ActivationFunc.SOFTMAX) 
    neural_model.fit(samples=X_train, target=y_train, num_epoch=30)
    accuracy_train = neural_model.calc_accuracy(X_train, y_train)
    accuracy_test = neural_model.calc_accuracy(X_test, y_test)
    print('Accuracy: train - %6.3f, test - %6.3f' % (accuracy_train, accuracy_test))
        