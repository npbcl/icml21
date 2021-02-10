import sys

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()


import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
try:
    os.mkdir('./Gens')
    os.mkdir('./saves')
except:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import math
from data_generators import OneMnistGenerator, OneNotMnistGenerator, OneFashionMnistGenerator
from ibpbcl_tfree import IBP_BCL



hidden_size = [500, 500, 100]
# alpha = [80.0, 80.0, 20.0, 80.0, 80.0]
alpha = [[80.0, 40.0, 20.0, 40.0, 80.0]]
no_epochs = 400#
no_tasks = 10
coreset_size = 0#50
coreset_method = "rand"
single_head = True
batch_size = 128


data_gen = OneMnistGenerator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head)


liks, _ = model.train_tfree(batch_size)
np.save('./saves/mnist_likelihoods.npy', liks)