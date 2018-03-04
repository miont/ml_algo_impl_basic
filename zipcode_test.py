# -*- coding: utf-8 -*-
"""
Задание Ex2.8 из книги The Elements of Statistical Learning

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

USE_LINEAR_MODEL = 0
USE_KNN = 1
USE_FEEDFORWARD_NN = 0
if __name__ == '__main__':
    
    print('Loadind train data')      
    data_train = read_data('data/zip.train')
    print('Loadind test data')      
    data_test = read_data('data/zip.test')
    
    print('Selecting train and test sets.')
    labels_used = (2,3)
    labels_used = (5,6)
    mask_train = (data_train[:,0] == labels_used[0]) | (data_train[:,0] == labels_used[1])
    X_train = data_train[mask_train][:,1:]
    y_train = data_train[mask_train][:,0]    
    y_train = map_labels(y_train, labels_used)
    print('%d examples in train set' % (X_train.shape[0]))
    mask_test = (data_test[:,0] == labels_used[0]) | (data_test[:,0] == labels_used[1])
    X_test = data_test[mask_test][:,1:]
    y_test = data_test[mask_test][:,0]
    y_test = map_labels(y_test, labels_used)
    print('%d examples in test set' % (X_test.shape[0]))
    num_train_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    
    if USE_LINEAR_MODEL:
        print('---Linear model---')
        print('Creating model')
        start_time = datetime.datetime.now()
        linear_model = LinearClassifier()
        print('Fitting model')
        accuracy_train = linear_model.fit(X_train, y_train)
        print('Evaluation on the test set')
        accuracy_test = linear_model.calc_accuracy(X_test, y_test)
        print('Accuracy: train - %5.2f, test - %5.2f' % (accuracy_train, accuracy_test))
        y_pred = map_labels(linear_model.predict(X_test),labels_used,reverse=True)
        print('Elapsed time:' + str(datetime.datetime.now() - start_time))
    
    if USE_KNN:
        print('---kNN---')   
        k_list = [1, 3, 5, 7, 15]
        var_count = len(k_list)
        knn_errors = {}
        knn_errors['train'] = np.zeros((var_count,1))
        knn_errors['test'] = np.zeros((var_count,1))
        for i,k in enumerate(k_list):
            print('k = ', k)
            start_time = datetime.datetime.now()
            print('Creating model')
            kNN_model = KNNClassifier(k)
            print('KNNClassifier.k = ', kNN_model.k)
            print('Fitting model')
            accuracy_train = kNN_model.fit(X_train, y_train)
            print('Evaluation on the test set')
            accuracy_test = kNN_model.calc_accuracy(X_test, y_test)
            print('Accuracy: train - %5.2f, test - %5.2f' % (accuracy_train, accuracy_test))
            y_pred = map_labels(linear_model.predict(X_test),labels_used,reverse=True)
            print('Elapsed time:' + str(datetime.datetime.now() - start_time))
            knn_errors['train'][i] = 1-accuracy_train
            knn_errors['test'][i] = 1-accuracy_test
        ind_min = np.argmin(knn_errors['test'])
        k_best = k_list[ind_min]
        max_accuracy_test = 1 - knn_errors['test'][ind_min]
        print('kNN: maximum accuracy on test set: %5.2f for k = %d', (max_accuracy_test,k_best))
            
        plt.close('all')
        plt.figure(1)
        plt.plot(k_list,knn_errors['train'], '-ob')
        plt.plot(k_list,knn_errors['test'], '-or')
        plt.legend(['train','test'])    
        plt.xlabel('k - Number of Nearest Neighbors')
        plt.ylabel('error')
        plt.grid(True, linestyle='-', color='0.75')
        plt.title('kNN errors')
        plt.show()
        
    if USE_FEEDFORWARD_NN:
        print('---Feedforward neural net---')
        net_config = (num_features,100,50,10,1)
#        net_config = (num_features,1)
        neural_model = FeedforwardNeuralNet(layers_size=net_config, activation=ActivationFunc.SIGMOID, output_func=ActivationFunc.SIGMOID)
        y_train_1 = np.array([1 if x==1 else 0 for x in y_train])        
        neural_model.fit(samples=X_train, target=y_train_1, num_epoch=10)
        accuracy_train = neural_model.calc_accuracy(X_train, y_train)
        accuracy_test = neural_model.calc_accuracy(X_test, y_test)
        print('Accuracy: train - %6.3f, test - %6.3f' % (accuracy_train, accuracy_test))
        