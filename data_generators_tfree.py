import os
import numpy as np
import math
from copy import deepcopy
import gzip
import pickle
import dataset_loader
from sklearn.model_selection import train_test_split

class OneMnistGenerator():
    def __init__(self):
        train_imgs, train_labels, test_imgs, test_labels = dataset_loader.load('mnist')
        train_imgs = train_imgs.reshape([-1,784])
        test_imgs = test_imgs.reshape([-1,784])
        self.X_train = train_imgs
        self.X_test = test_imgs
        self.train_label = train_labels
        self.test_label = test_labels

        self.sets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.max_iter = len(self.sets)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.X_train.shape[1]

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            train_id = np.where(self.train_label == self.sets[self.cur_iter])[0]
            next_x_train = self.X_train[train_id]
            next_y_train = np.ones((train_id.shape[0], 1))

            # Retrieve test data
            
            test_id = np.where(self.test_label == self.sets[self.cur_iter])[0]
            next_x_test = self.X_test[test_id]
            next_y_test = np.ones((test_id.shape[0], 1))

            self.cur_iter += 1

            return next_x_train, next_x_train, next_x_test, next_x_test


class OneNotMnistGenerator():
    def __init__(self):
        train_imgs, train_labels, test_imgs, test_labels = dataset_loader.load('notmnist')
        train_imgs = train_imgs.reshape([-1,784])
        test_imgs = test_imgs.reshape([-1,784])
        self.X_train = train_imgs
        self.X_test = test_imgs
        self.train_label = train_labels
        self.test_label = test_labels


        self.sets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.max_iter = len(self.sets)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.X_train.shape[1]

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            train_id = np.where(self.train_label == self.sets[self.cur_iter])[0]
            next_x_train = self.X_train[train_id]
            next_y_train = np.ones((train_id.shape[0], 1))

            # Retrieve test data
            
            test_id = np.where(self.test_label == self.sets[self.cur_iter])[0]
            next_x_test = self.X_test[test_id]
            next_y_test = np.ones((test_id.shape[0], 1))

            self.cur_iter += 1
            # print(next_x_test.shape)
            # assert 1 == 2
            return next_x_train/255, next_x_train/255, next_x_test/255, next_x_test/255
        
        
class OneFashionMnistGenerator():
    def __init__(self):
        train_imgs, train_labels, test_imgs, test_labels = dataset_loader.load('fashionmnist')
        train_imgs = train_imgs.reshape([-1,784])
        test_imgs = test_imgs.reshape([-1,784])
        self.X_train = train_imgs
        self.X_test = test_imgs
        self.train_label = train_labels
        self.test_label = test_labels


        self.sets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.max_iter = len(self.sets)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.X_train.shape[1]

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            train_id = np.where(self.train_label == self.sets[self.cur_iter])[0]
            next_x_train = self.X_train[train_id]
            next_y_train = np.ones((train_id.shape[0], 1))

            # Retrieve test data
            
            test_id = np.where(self.test_label == self.sets[self.cur_iter])[0]
            next_x_test = self.X_test[test_id]
            next_y_test = np.ones((test_id.shape[0], 1))

            self.cur_iter += 1
            # print(next_x_test.shape)
            # assert 1 == 2
            return next_x_train/255, next_x_train/255, next_x_test/255, next_x_test/255