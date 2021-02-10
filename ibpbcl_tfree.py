import os
from ibpbnn_tfree import IBP_BAE
import copy as cpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.distributions as tod
import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import gzip
import pickle

class IBP_BCL:
    def __init__(self, hidden_size, alpha, no_epochs, data_gen, coreset_method, coreset_size=0, single_head=True):
        '''
        hidden_size : list of network hidden layer sizes
        alpha : IBP prior concentration parameters
        data_gen : Data Generator
        coreset_size : Size of coreset to be used (0 represents no coreset)
        single_head : To given single head output for all task or multihead output for each task seperately.
        '''
        ## Intializing Hyperparameters for the model.
        self.hidden_size = hidden_size
        self.alpha = alpha#[alpha for i in range(len(hidden_size)*2-1)]
        self.beta = [[1.0 for i in range(len(hidden_size)*2-1)]]
        self.no_epochs = no_epochs
        self.data_gen = data_gen


        self.Dnew = 1000 # size of minmum shortterm for new cluster creation.
        self.first_cluster_init = False # If first cluster has been trained or not atleast once.

        if(coreset_method != "kcen"):
            self.coreset_method = self.rand_from_batch
        else:
            self.coreset_method = self.k_center
        self.coreset_size = coreset_size
        self.single_head = single_head 
        self.cuda = torch.cuda.is_available()
    
    def rand_from_batch(self, x_coreset, y_coreset, x_train, y_train, coreset_size):
        """ Random coreset selection """
        # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        idx = np.random.choice(x_train.shape[0], coreset_size, False)
        x_coreset.append(x_train[idx,:])
        y_coreset.append(y_train[idx,:])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        return x_coreset, y_coreset, x_train, y_train    

    def k_center(self, x_coreset, y_coreset, x_train, y_train, coreset_size):
        """ K-center coreset selection """
        # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
        dists = np.full(x_train.shape[0], np.inf)
        current_id = 0
        dists = self.update_distance(dists, x_train, current_id)
        idx = [current_id]
        for i in range(1, coreset_size):
            current_id = np.argmax(dists)
            dists = update_distance(dists, x_train, current_id)
            idx.append(current_id)
        x_coreset.append(x_train[idx,:])
        y_coreset.append(y_train[idx,:])
        x_train = np.delete(x_train, idx, axis=0)
        y_train = np.delete(y_train, idx, axis=0)
        return x_coreset, y_coreset, x_train, y_train

    def update_distance(self, dists, x_train, current_id):
        for i in range(x_train.shape[0]):
            current_dist = np.linalg.norm(x_train[i,:]-x_train[current_id,:])
            dists[i] = np.minimum(current_dist, dists[i])
        return dists
    
    def merge_coresets(self, x_coresets, y_coresets):
        ## Merges the current task coreset to rest of the coresets
        merged_x, merged_y = x_coresets[0], y_coresets[0]
        for i in range(1, len(x_coresets)):
            merged_x = np.vstack((merged_x, x_coresets[i]))
            merged_y = np.vstack((merged_y, y_coresets[i]))
        return merged_x, merged_y
    
    def logit(self, x):
        eps = 10e-8
        return (np.log(x+eps) - np.log(1-x+eps))
    
    def get_soft_logit(self, masks, task_id):
        var = []
        for i in range(len(masks)):
            var.append(self.logit(masks[i][task_id]*0.8 + 0.1))
        
        return var
       
    def get_scores(self, model, x_testsets, y_testsets, x_coresets, y_coresets, hidden_size, 
                   no_epochs, single_head, batch_size=None, kl_mask = None):
        ## Retrieving the current model parameters
        mf_model = model
        mf_weights, mf_variances = model.get_weights()
        prev_masks, self.alpha, self.beta = mf_model.get_IBP()
        logliks = []
        
        ## In case the model is single head or have coresets then we need to test accodingly.
        if single_head:# If model is single headed.
            if len(x_coresets) > 0:# Model has non zero coreset size
                del mf_model
                torch.cuda.empty_cache() 
                x_train, y_train = self.merge_coresets(x_coresets, y_coresets)
                prev_pber = self.get_soft_logit(prev_masks,i)
                bsize = x_train.shape[0] if (batch_size is None) else batch_size
                final_model = IBP_BAE(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], self.max_tasks,
                                   prev_means=mf_weights, prev_log_variances=mf_variances, 
                                   prev_masks = prev_masks, alpha=alpha, beta = beta, prev_pber = prev_pber, 
                                   kl_mask = kl_mask, single_head=single_head)
                final_model.ukm = 1
                final_model.batch_train(x_train, y_train, 0, self.no_epochs, bsize, max(self.no_epochs//5,1))
            else:# Model does not have coreset
                final_model = model

        ## Testing for all previously learned tasks
        num_samples = 10
        fig, ax = plt.subplots(num_samples, len(x_testsets), figsize = [10,10])
        for i in range(len(x_testsets)):
            if not single_head:# If model is multi headed.
                if len(x_coresets) > 0:
                    try:
                        del mf_model
                    except:
                        pass
                    torch.cuda.empty_cache() 
                    x_train, y_train = x_coresets[i], y_coresets[i]# coresets per task
                    prev_pber = self.get_soft_logit(prev_masks,i)
                    bsize = x_train.shape[0] if (batch_size is None) else batch_size
                    final_model = IBP_BAE(x_train.shape[1], hidden_size, y_train.shape[1], x_train.shape[0], self.max_tasks, 
                                   prev_means=mf_weights, prev_log_variances=mf_variances, 
                                   prev_masks = prev_masks, alpha=alpha, beta = beta, prev_pber = prev_pber, 
                                   kl_mask = kl_mask, learning_rate = 0.0001, single_head=single_head)
                    final_model.ukm = 1
                    final_model.batch_train(x_train, y_train, i, self.no_epochs, bsize, max(self.no_epochs//5,1), init_temp = 0.25)
                else:
                    final_model = model
            
            
            x_test, y_test = x_testsets[i], y_testsets[i]
            # pred = final_model.prediction_prob(x_test, i)
            pred = self.model.prediction_prob(x_test, i)
            pred_mean = np.mean(pred, axis=1) # N x O
            eps = 10e-8
            target = y_test#targets.unsqueeze(1).repeat(1, self.no_train_samples, 1)# Formating desired output : N x O
            loss = np.sum(- target * np.log(pred_mean+eps) - (1.0 - target) * np.log(1.0-pred_mean+eps) , axis = -1)
            log_lik = - (loss).mean()# Binary Crossentropy Loss
            logliks.append(log_lik)

            # samples = pred_mean[:num_samples]
            samples = final_model.gen_samples(i, num_samples).cpu().detach().numpy()
            recosn = pred_mean[:num_samples]
            for s in range(num_samples):
                if(len(x_testsets) == 1):
                    ax[s].imshow(np.reshape(recosn[s], [28,28]))
                else:
                    ax[s][i].imshow(np.reshape(recosn[s], [28,28]))

        plt.savefig('./Gens/Task_till_' + str(i) +'.png')

        return logliks

    def concatenate_results(self, score, all_score):
        ## Concats the current accuracies on all task to previous result in form of matrix
        if all_score.size == 0:
            all_score = np.reshape(score, (1,-1))
        else:
            new_arr = np.empty((all_score.shape[0], all_score.shape[1]+1))
            new_arr[:] = np.nan# Puts nan in place of empty values (tasks that previous model was not trained on)
            new_arr[:,:-1] = all_score
            all_score = np.vstack((new_arr, score))
        return all_score
        
    def batch_train(self, in_dim, out_dim, training_size, prev_means, prev_log_variances, prev_masks , prev_pber, kl_mask, x_train, y_train, bsize):

            ## Training the network  
            # mf_model = IBP_BAE(in_dim, self.hidden_size, out_dim, training_size, self.max_tasks, 
            #                    prev_means=prev_means, prev_log_variances=prev_log_variances, 
            #                    prev_masks = prev_masks, alpha=self.alpha, beta = self.beta, prev_pber = prev_pber, 
            #                    kl_mask = kl_mask, single_head=self.single_head, extend = False)
            if(self.model is None):
                print("No Model Exists...")
                self.get_new_model(in_dim, out_dim, training_size, prev_means=prev_means, prev_log_variances=prev_log_variances, 
                                prev_masks = prev_masks, prev_pber = prev_pber, kl_mask = kl_mask, extend = False)
                
            # First we need a batch size enough to train for first task
            # Then we need to check if we need expansion so we will go with likelihood thresholding.
            cur_index = self.Dnew
            # Step 1. Getting Likelihoods for all current tasks
            n_epochs_per_batch = self.no_epochs # int(np.ceil(self.no_epochs*self.Dnew/x_train.shape[0]))
            if(not self.first_cluster_init): # If netwrok has never been trained.
                x_train_batch = x_train[:self.Dnew]
                y_train_batch = y_train[:self.Dnew]
                cur_index = self.Dnew
                self.model.batch_train(x_train_batch, y_train_batch, 0, no_epochs = n_epochs_per_batch, batch_size = bsize, display_epoch = max(n_epochs_per_batch//5,1))
                self.first_cluster_init = True
                
            new_x = None
            new_y = None
            while(cur_index < training_size):
                print("\n\nStream Complete : {}/{}\n\n".format(cur_index, training_size))
                x_train_batch = x_train[cur_index:cur_index + self.Dnew]
                y_train_batch = y_train[cur_index:cur_index + self.Dnew]
                cur_index += self.Dnew
                task_ll = self.model.get_loglikeli(x_train_batch, y_train_batch).cpu().view(-1).detach().numpy()
                X_new = x_train_batch[task_ll > -160,:]
                Y_new = y_train_batch[task_ll > -160,:]
                if(new_x is None):
                    new_x = X_new
                    new_y = Y_new
                else:
                    new_x = np.concatenate([new_x,X_new], axis = 0)
                    new_y = np.concatenate([new_y.reshape([-1,1]),Y_new.reshape([-1,1])], axis = 0)
                
                x_train_batch = x_train_batch[task_ll < -160]
                y_train_batch = y_train_batch[task_ll < -160]

                self.model.batch_train(x_train_batch, y_train_batch, 0, no_epochs = n_epochs_per_batch, batch_size = bsize, display_epoch = max(n_epochs_per_batch//5,1))
                    
                print("current lenghth :", len(new_x))
                if(len(new_x) >= self.Dnew):

                    mf_weights, mf_variances = self.model.get_weights()
                    prev_masks, self.alpha, self.beta = self.model.get_IBP()

                    self.get_new_model(in_dim, out_dim, training_size, prev_means=prev_means, prev_log_variances=prev_log_variances, 
                               prev_masks = prev_masks, prev_pber = prev_pber, kl_mask = kl_mask, extend = False)

                    # mf_model = IBP_BAE(in_dim, self.hidden_size, out_dim, training_size, self.max_tasks, 
                    #                 prev_means=prev_means, prev_log_variances=prev_log_variances, 
                    #                 prev_masks = prev_masks, alpha=self.alpha, beta = self.beta, prev_pber = prev_pber, 
                    #                 kl_mask = kl_mask, single_head=self.single_head, extend = True)
                    
                    self.model.batch_train(new_x, new_y, 0, no_epochs = n_epochs_per_batch, batch_size = bsize, display_epoch = max(n_epochs_per_batch//5,1))

                    new_x = None
                    new_y = None


                # if torch.cuda.device_count() > 1: 
                #     mf_model = nn.DataParallel(mf_model) #enabling data parallelism

            

            mf_weights, mf_variances = mf_model.get_weights()
            prev_masks, self.alpha, self.beta = mf_model.get_IBP()
            return mf_weights, mf_variances, prev_masks, mf_model

    def get_new_model(self, in_dim, out_dim, training_size, prev_means, prev_log_variances, 
                prev_masks, prev_pber, kl_mask, extend):
        
        if(self.model is None):
            print("Creating Initial Model Instance")
            self.model = IBP_BAE(in_dim, self.hidden_size, out_dim, training_size, self.max_tasks, 
                               prev_means=prev_means, prev_log_variances=prev_log_variances, 
                               prev_masks = prev_masks, alpha=self.alpha, beta = self.beta, prev_pber = prev_pber, 
                               kl_mask = kl_mask, single_head=self.single_head, extend = extend)
        else:
            print("Recreating Model Instance")
            ## Calculating Union of all task masks and also for visualizing the layer wise network sparsity
            sparsity = []
            kl_mask = []
            M = len(mf_variances[0])
            for j in range(M):
                ## Plotting union mask
                var = (np.sum(prev_masks[j][:task_id+1],0)>0.5)*1.02
                mask = (var > 0.5)*1
                mask2 = (np.sum(prev_masks[j][:task_id+1],0) > 0.1)*1.0
                ## Calculating network sparsity
                var2 = (np.sum(prev_masks[j][:task_id+1],0) > 0.5)
                kl_mask.append(var2)
                filled = np.mean(mask)
                sparsity.append(filled)
            
            # ax1[task_id].imshow(mask2,vmin=0, vmax=1)
            # fig1.savefig("union_mask.png")
            print("Network sparsity : ", sparsity)
            del self.model
            torch.cuda.empty_cache() 
            
            self.model = IBP_BAE(in_dim, self.hidden_size, out_dim, training_size, self.max_tasks, 
                               prev_means=prev_means, prev_log_variances=prev_log_variances, 
                               prev_masks = prev_masks, alpha=self.alpha, beta = self.beta, prev_pber = prev_pber, 
                               kl_mask = kl_mask, single_head=self.single_head, extend = extend)

        if(self.cuda):
            print("Cuda Exists : Transfering model on GPU")
            model = self.model
            self.model = model.to('cuda')#.cuda()
            self.model.device = 'cuda'
            print('Done')
        print("Model Creation Complete")
        

    
    def train_tfree(self, batch_size = None):

        '''
        batch_size : Batch_size for gradient updates
        '''
        print("\n\n Entering Training Phase")
        np.set_printoptions(linewidth=np.inf)
        ## Intializing coresets and dimensions.
        in_dim, out_dim = self.data_gen.get_dims()
        x_coresets, y_coresets = [], []
        x_testsets, y_testsets = [], []
        x_trainset, y_trainset = [], []
        all_acc = np.array([])
        self.max_tasks = self.data_gen.max_iter
        self.model = None
        # fig1, ax1 = plt.subplots(1,self.max_tasks, figsize = [10,5])
        ## Training the model sequentially.
        for task_id in range(self.max_tasks):
            ## Loading training and test data for current task
            x_train, y_train, x_test, y_test = self.data_gen.next_task()
            x_testsets.append(x_test)
            y_testsets.append(y_test)
            ## Initializing the batch size for training 
            bsize = x_train.shape[0] if (batch_size is None) else batch_size
            ## If this is the first task we need to initialize few variables.
            if task_id == 0:
                prev_masks = None
                prev_pber = None
                kl_mask = None
                mf_weights = None
                mf_variances = None
            ## Select coreset if coreset size is non zero
            if self.coreset_size > 0:
                x_coresets,y_coresets,x_train,y_train = self.coreset_method(x_coresets,y_coresets,x_train,y_train,self.coreset_size)

            self.batch_train(in_dim, out_dim, x_train.shape[0], mf_weights, mf_variances, prev_masks , prev_pber, kl_mask, x_train, y_train, bsize)
            
            mf_weights, mf_variances = self.model.get_weights()
            prev_masks, self.alpha, self.beta = self.model.get_IBP()
            ## Calculating Union of all task masks and also for visualizing the layer wise network sparsity
            sparsity = []
            kl_mask = []
            M = len(mf_variances[0])
            for j in range(M):
                ## Plotting union mask
                var = (np.sum(prev_masks[j][:task_id+1],0)>0.5)*1.02
                mask = (var > 0.5)*1
                mask2 = (np.sum(prev_masks[j][:task_id+1],0) > 0.1)*1.0
                ## Calculating network sparsity
                var2 = (np.sum(prev_masks[j][:task_id+1],0) > 0.5)
                kl_mask.append(var2)
                filled = np.mean(mask)
                sparsity.append(filled)
            
            # ax1[task_id].imshow(mask2,vmin=0, vmax=1)
            # fig1.savefig("union_mask.png")
            print("Network sparsity : ", sparsity)
            acc = self.get_scores(mf_model, x_testsets, y_testsets, x_coresets, y_coresets, 
                                  self.hidden_size, self.no_epochs, self.single_head, batch_size, kl_mask)

            torch.save(mf_model.state_dict(), "./saves/model_last_" + str(task_id))
            del mf_model
            torch.cuda.empty_cache() 
            all_acc = self.concatenate_results(acc, all_acc); print(all_acc.round(3)); print('*****')

        np.savetxt('./Gens/res.txt', all_acc)
        return [all_acc, prev_masks]