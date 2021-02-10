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
try:
    os.mkdir('./saves')
except:
    pass
import numpy as np
import matplotlib
matplotlib.use('Agg')
import math
from data_generators import PermutedMnistGenerator, SplitMnistGenerator, NotMnistGenerator, FashionMnistGenerator, SplitCifar100Generator10, SplitCifar100Generator, SplitCifar10Generator5
from ibpbcl import IBP_BCL
import torch


torch.manual_seed(8)
np.random.seed(10)


# hidden_size = [64,256,'512',256, 128]
# alpha = [8,32,64,32,16]
hidden_size = [128,256,512,'2048',2048, 2048]
alpha = [20,60,160,440,440,440]
no_epochs = 10
no_tasks = 5
coreset_size = 0#200
coreset_method = "kcen"
single_head = False
batch_size = 250

# data_gen = PermutedMnistGenerator(no_tasks)
# data_gen = SplitMnistGenerator()
# data_gen = NotMnistGenerator()
# data_gen = FashionMnistGenerator()
# data_gen = SplitCifar100Generator10()
data_gen = SplitCifar100Generator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head, grow = False)

accs, _ = model.batch_train(batch_size)
np.save('./saves/permutedmnist_accuracies.npy', accs)
