import os
import numpy as np
import math
from copy import deepcopy
import gzip
import pickle
import dataset_loader
from sklearn.model_selection import train_test_split as tts
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
import matplotlib.pyplot as plt
plt.rcParams["axes.grid"] = False

class PermutedMnistGenerator():
    def __init__(self, max_iter=10):
        train_imgs, train_labels, test_imgs, test_labels = dataset_loader.load('mnist')
        train_imgs = train_imgs.reshape([-1,784])
        test_imgs = test_imgs.reshape([-1,784])
        self.X_train = train_imgs
        self.Y_train = train_labels
        self.X_test = test_imgs
        self.Y_test = test_labels
        self.max_iter = max_iter
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            np.random.seed(self.cur_iter)
            perm_inds = np.arange(self.X_train.shape[1])
            np.random.shuffle(perm_inds)

            # Retrieve train data
            next_x_train = deepcopy(self.X_train)
            next_x_train = next_x_train[:,perm_inds]
            next_y_train = np.eye(10)[self.Y_train]

            # Retrieve test data
            next_x_test = deepcopy(self.X_test)
            next_x_test = next_x_test[:,perm_inds]
            next_y_test = np.eye(10)[self.Y_test]

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SplitMnistGenerator():
    def __init__(self):
        train_imgs, train_labels, test_imgs, test_labels = dataset_loader.load('mnist')
        train_imgs = train_imgs.reshape([-1,784])
        test_imgs = test_imgs.reshape([-1,784])
        self.X_train = train_imgs
        self.X_test = test_imgs
        self.train_label = train_labels
        self.test_label = test_labels

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SplitCifar100Generator():
    def __init__(self):
        self.tasks = dataset_loader.load('cifar100')
        self.cur_iter = 0
        self.max_iter = len(self.tasks)
    def get_dims(self):
        # Get data input and output dimensions
        return 3, 5

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            next_x_train, next_y_train, next_x_test, next_y_test = self.tasks[self.cur_iter]
            # print(next_x_train.shape)
            next_x_train = np.transpose(next_x_train, [0,3,1,2])
            next_x_test = np.transpose(next_x_test, [0,3,1,2])
            # print(next_x_train.shape)
            # assert False
            mean = np.array([x/255 for x in [125.3,123.0,113.9]]).reshape(1,-1,1,1)
            std = np.array([x/255 for x in [63.0,62.1,66.7]]).reshape(1,-1,1,1)

            next_x_train = (next_x_train.reshape([-1,3,32,32])/255.0 - mean)/std
            next_x_test = (next_x_test.reshape([-1,3,32,32])/255.0 - mean)/std

            vals = np.unique(next_y_train)
            next_y_tr = np.zeros_like(next_y_train)
            next_y_te = np.zeros_like(next_y_test)
            for i,k in enumerate(vals):
                next_y_tr[np.where(next_y_train == k)] = i
                next_y_te[np.where(next_y_test == k)] = i


            next_y_train = to_categorical(next_y_tr, num_classes=5)
            next_y_test = to_categorical(next_y_te, num_classes=5)
            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

class SplitCifar100Generator10():
    def __init__(self):
        self.tasks = dataset_loader.load('cifar100')
        self.cur_iter = 0
        self.max_iter = len(self.tasks)//2
    def get_dims(self):
        # Get data input and output dimensions
        return 3, 10

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            next_x_train1, next_y_train1, next_x_test1, next_y_test1 = self.tasks[self.cur_iter]
            next_x_train2, next_y_train2, next_x_test2, next_y_test2 = self.tasks[self.cur_iter+self.max_iter]
            next_x = np.concatenate([next_x_train1, next_x_train2, next_x_test1, next_x_test2], 0)
            next_y = np.concatenate([next_y_train1, next_y_train2, next_y_test1, next_y_test2], 0)
            # print(next_x.shape, next_x.max())
            # assert False
            
            next_x = np.transpose(next_x, [0,3,1,2])
            # next_x = np.transpose(next_x, [0,3,1,2])
            # print(next_x_train.shape)
            # assert False

            mean = np.array([x/255 for x in [125.3,123.0,113.9]]).reshape(1,-1,1,1)
            std = np.array([x/255 for x in [63.0,62.1,66.7]]).reshape(1,-1,1,1)
            next_x = (next_x.reshape([-1,3,32,32])/255.0 - mean)/std
            
            # next_x_test = (next_x_test.reshape([-1,3,32,32])/255.0 - mean)/std

            mid = next_x_train1.shape[0] + next_x_train2.shape[0]
            next_x_train = next_x[:mid]
            next_x_test = next_x[mid:]
            next_y_train = next_y[:mid]
            next_y_test = next_y[mid:]

            # next_x_train, next_x_test, next_y_train, next_y_test = tts(next_x, next_y, test_size=0.05, stratify = next_y)
            
            # next_x_train = np.transpose(next_x_train.reshape([-1,3,32,32]), [0,2,3,1])
            # next_x_test = np.transpose(next_x_test.reshape([-1,3,32,32]), [0,2,3,1])
            # next_x_train = next_x_train.reshape([-1,3,32,32])
            # next_x_test = next_x_test.reshape([-1,3,32,32])

            vals = np.unique(next_y_train)
            next_y_tr = np.zeros_like(next_y_train)
            next_y_te = np.zeros_like(next_y_test)
            # fig, ax = plt.subplots(5, 10)
            # plt.grid(None) 
            for i,k in enumerate(vals):
                next_y_tr[np.where(next_y_train == k)] = i
                next_y_te[np.where(next_y_test == k)] = i

                # ax[0][i].set_title(str(i))
                # y_s = np.where(next_y == k)
            #     curr = next_x[y_s]
            #     for s in range(5):
            #         ax[s][i].imshow(np.transpose(curr[np.random.randint(len(curr))], [1,2,0]))
            # plt.savefig("vis.png")
            # assert False
            
            next_y_train = to_categorical(next_y_tr, num_classes=10)
            next_y_test = to_categorical(next_y_te, num_classes=10)
            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
        
class SplitCifar10Generator5():
    def __init__(self):
        self.tasks = dataset_loader.load('cifar10')
        self.cur_iter = 0
        self.max_iter = len(self.tasks)
    def get_dims(self):
        # Get data input and output dimensions
        return 3, 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            next_x_train, next_y_train, next_x_test, next_y_test = self.tasks[self.cur_iter]
            next_x_train = np.transpose(next_x_train, [0,3,1,2])
            next_x_test = np.transpose(next_x_test, [0,3,1,2])
            # print(next_x_train.shape)
            # next_x_train = next_x_train.reshape([-1,3,32,32])
            # next_x_test = next_x_test.reshape([-1,3,32,32])
            # print(next_x_train.shape)
            # assert False
            mean = np.array([x/255 for x in [125.3,123.0,113.9]]).reshape(1,-1,1,1)
            std = np.array([x/255 for x in [63.0,62.1,66.7]]).reshape(1,-1,1,1)

            next_x_train = (next_x_train.reshape([-1,3,32,32])/255.0 - mean)/std
            next_x_test = (next_x_test.reshape([-1,3,32,32])/255.0 - mean)/std

            vals = np.unique(next_y_train)
            next_y_tr = np.zeros_like(next_y_train)
            next_y_te = np.zeros_like(next_y_test)
            for i,k in enumerate(vals):
                next_y_tr[np.where(next_y_train == k)] = i
                next_y_te[np.where(next_y_test == k)] = i


            next_y_train = to_categorical(next_y_tr, num_classes=2)
            next_y_test = to_categorical(next_y_te, num_classes=2)
            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test
    
class NotMnistGenerator():
    def __init__(self):
        train_imgs, train_labels, test_imgs, test_labels = dataset_loader.load('notmnist')
        train_imgs = train_imgs.reshape([-1,784])
        test_imgs = test_imgs.reshape([-1,784])
        self.X_train = train_imgs
        self.X_test = test_imgs
        self.train_label = train_labels
        self.test_label = test_labels

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train/255, next_y_train, next_x_test/255, next_y_test

class FashionMnistGenerator():
    def __init__(self):
        train_imgs, train_Y, test_imgs, test_Y = dataset_loader.load('fashionmnist')
        train_X = train_imgs.reshape([-1,784])
        test_X = test_imgs.reshape([-1,784])

        self.X_train = train_X.reshape([-1, 784])/255
        self.X_test = test_X.reshape([-1, 784])/255
        self.train_label = train_Y
        self.test_label = test_Y

        self.sets_0 = [0, 2, 4, 6, 8]
        self.sets_1 = [1, 3, 5, 7, 9]
        self.max_iter = len(self.sets_0)
        self.cur_iter = 0

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], 2

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            self.cur_iter = 0
            raise Exception('Number of tasks exceeded! Now resetted ! Try again')
        else:
            # Retrieve train data
            train_0_id = np.where(self.train_label == self.sets_0[self.cur_iter])[0]
            train_1_id = np.where(self.train_label == self.sets_1[self.cur_iter])[0]
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))

            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))

            # Retrieve test data
            test_0_id = np.where(self.test_label == self.sets_0[self.cur_iter])[0]
            test_1_id = np.where(self.test_label == self.sets_1[self.cur_iter])[0]
            next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))

            next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
            next_y_test = np.hstack((next_y_test, 1-next_y_test))

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

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