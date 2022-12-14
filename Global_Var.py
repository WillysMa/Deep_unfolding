# -*- oding:utf-8 -*-
'''
# @File: Global_Var.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2022-12-14 1:38 PM
'''
import math
import os
# System parameters
Nt = 8
Nr = 4
Nrf = Nr
Ns = Nrf
N = 2 * Nt * Nrf
Ncl = 1  ## number of clusters
Nray = 4 ## number of rays in each cluster
GHz = 1e+9
K = 5
Bandwidth = 30 * GHz  # system bandwidth
fc = 300 * GHz   # carrier frequency
Array_Type = 'UPA'
init_scheme = 0  # 0 for random initialization or 2 fro OMP initialization
Num_layers = math.ceil(math.log2(Nt))
Sub_Connected = False
# Sub_Structure_Type = 'fixed'
Sub_Structure_Type = 'dyn'

if init_scheme:
    train_data_name = 'train_set.hdf5'
else:
    train_data_name = 'train_set_RDM.hdf5'
GenNum_Batch_tr = 10  # used for generating training data
Gen_Batch_size_tr = 10  # used for generating training data
training_set_size = GenNum_Batch_tr * Gen_Batch_size_tr

if init_scheme:
    test_data_name = 'test_set.hdf5'
else:
    test_data_name = 'test_set_RDM.hdf5'
GenNum_Batch_te = 5  # used for generating testing data
Gen_Batch_size_te = 10  # used for generating testing data
testing_set_size = GenNum_Batch_te * Gen_Batch_size_te
test_batch_size = 50

# Training parameters
Seed_train = 1
Seed_test = 101

training_set_size_truncated = int(Num_layers*100)  # the number of data used in the training stage
train_batch_size = 10
Ntrain_batch_total = 100 # total number of training batches
Ntrain_Batch_perEpoch = training_set_size_truncated // train_batch_size  # number of batches in per epoch
Ntrain_Epoch = math.ceil(Ntrain_batch_total/Ntrain_Batch_perEpoch)  # number of training epoch



training_method = 'unsupervised'
Iterative_Training = False
Iterations_train = 3

subfile = 'ManNet/'
if Sub_Connected:
    if Sub_Structure_Type == 'fixed':
        subsubfile = 'PC_HB_Fixed/'
    else:
        subsubfile = 'PC_HB_Dyn/'
else:
    subsubfile = 'FC-HB/'

if Iterative_Training:
    sssbfile = 'Iterative_train/'
else:
    sssbfile = 'Non-iterative_train/'

dataset_file = "./trained_model/" + str(Nt) + "x" + str(Nr) + "x" + str(Nrf) + "x" + str(K) + "/"
directory_model = dataset_file + subfile + subsubfile + sssbfile

model_file_name = directory_model + "trained_model"
if not os.path.exists(directory_model):
    os.makedirs(directory_model)