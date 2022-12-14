# -*- oding:utf-8 -*-
'''
# @File: main.py
# @Author: Mengyuan Ma
# @Contact: mamengyuan410@gmail.com
# @Time: 2022-12-14 1:35 PM
'''
from Gen_Data import *
from Global_Vars import *

dataloader_tr = Data_Fetch(file_dir=dataset_file,
                           file_name=train_data_name,
                           batch_size=train_batch_size,
                           training_set_size=training_set_size,
                           training_set_size_truncated=training_set_size_truncated,
                           data_str='training')
dataloader_te = Data_Fetch(file_dir=dataset_file,
                           file_name=test_data_name,
                           batch_size=test_batch_size,
                           training_set_size=testing_set_size,
                           data_str='testing')

batch_count = 0
for epoch in range(Ntrain_Epoch):
    dataloader_tr.reset()
    print('-----------------------------------------------')
    for batch_idx in range(Ntrain_Batch_perEpoch):
        batch_count += 1
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt = dataloader_tr.get_item()
        print(f'Epoch:{epoch}, batch_id:{batch_count}', flush=True)
        '''
        training here
        '''