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
from ibpbcl_vae import IBP_BCL



hidden_size = [500, 500, 100]
# alpha = [80.0, 80.0, 20.0, 80.0, 80.0]
alpha = [40.0, 40.0, 20.0, 40.0, 40.0]
no_epochs = 1#30#
no_tasks = 10
coreset_size = 0#50
coreset_method = "rand"
single_head = True
batch_size = 128


data_gen = OneMnistGenerator()
model = IBP_BCL(hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size, single_head)
model.batch_train(batch_size)