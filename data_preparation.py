#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 10:45:17 2021

@author: 22905553
"""
import numpy as np
import pickle

action = 60
split = 'val'
evals = 'ntu/xview/'
data_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/NTU_RGBD_dataset/NTU_preprocessed_data_by_MSG3D/'+evals+split+'_data_joint.npy'
label_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/NTU_RGBD_dataset/NTU_preprocessed_data_by_MSG3D/'+evals+split+'_label.pkl'
out_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/imbalanced/'
out_path = out_path + evals

data = np.load(data_path, mmap_mode='r')


with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')
    
    


c = []
for k in range(action):
    s = 0
    for i in range(len(label)):
        if label[i] == k:
            s+=1
    c.append(s)

c = []
for k in range(2):
    s = 0
    for i in range(len(fall_label)):
        if fall_label[i] == k:
            s+=1
    c.append(s)
    
fall_label = []
fall_paths = []
fall_data = []

bad_action = []
import random
for k in range(action):
    if k in bad_action:
        continue
    else:        
        index = [i for i, e in enumerate(label) if e == k]
        '''
        if k!=42:        
            index = random.choices(index, k=6)
        '''
        for i in index:
            fall_data.append(data[i,:,:,:,:])
            fall_paths.append(sample_name[i])
            if k==42:
                fall_label.append(1)
            else:
                fall_label.append(0)

fall_data = np.stack(fall_data)        

with open(f'{out_path}/{split}_label.pkl', 'wb') as f:
    pickle.dump((fall_paths, list(fall_label)), f)           
        
np.save(out_path+'{}_joint.npy'.format(split), fall_data)      
  
'''      
split = 'val'
evals = 'ntu120/xsub/'
path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/imbalanced/'
path = path + evals
data_path = path+split+'_joint.npy'
label_path = path+split+'_label.pkl'

data = np.load(data_path, mmap_mode='r')


with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')
    
    
x = torch.tensor(data)
import torch
dd = torch.isnan(x)                
if True in dd:
    print('Problem in x')
'''

## UWA3D
import numpy as np
import pickle

action = 30
eval_='train'
split = 'train'
o_split = 'val3'
data_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+split+'_joint.npy'
label_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+split+'_label.pkl'
out_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/uwa3d/'+o_split


data = np.load(data_path, mmap_mode='r')
data=data/1000
#np.save('/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+'mod_data.npy', data)

with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')
    
    


c = []
for k in range(action):
    s = 0
    for i in range(len(label)):
        if label[i] == k:
            s+=1
    c.append(s)

c = []
for k in range(2):
    s = 0
    for i in range(len(fall_label)):
        if fall_label[i] == k:
            s+=1
    c.append(s)
    
fall_label = []
fall_paths = []
fall_data = []

bad_action = []
import random
for k in range(action):
    if k in bad_action:
        continue
    else:        
        index = [i for i, e in enumerate(label) if e == k]
        '''
        if k!=7:        
            index = random.choices(index, k=1)
        '''            
        for i in index:
            fall_data.append(data[i,:,:,:,:])
            fall_paths.append(sample_name[i])
            if k==7:
                fall_label.append(1)
            else:
                fall_label.append(0)

fall_data = np.stack(fall_data)        


with open(f'{out_path}/{eval_}_label.pkl', 'wb') as f:
    pickle.dump((fall_paths, list(fall_label)), f)           
        
np.save(out_path+'/{}_joint.npy'.format(eval_), fall_data)      
        
        
       



## NTU+TST V2 =========================================================  
# cd /home/uniwa/students3/students/22905553/linux/phd_codes/Fall_detection
import numpy as np
import pickle


split = 'val'
evals = 'ntu120/xsub/'
path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/imbalanced/'+ evals
data_path = path+split+'_joint.npy'
label_path = path+split+'_label.pkl'


data = np.load(data_path, mmap_mode='r')
#data=data/1000
#np.save('/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+'mod_data.npy', data)

with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')
    
       
fall_label = []
fall_paths = []
fall_data = []

for i, l in enumerate(label):
    #print(i)
    fall_data.append(data[i,:,:,:,:])
    fall_paths.append(sample_name[i])
    fall_label.append(l)
    
split = 'val'
evals = 'xsub/'
path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/TST_V2/'+ evals
data_path = path+split+'_joint.npy'
label_path = path+split+'_label.pkl'


tst_data = np.load(data_path, mmap_mode='r')
#data=data/1000
#np.save('/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+'mod_data.npy', data)

with open(label_path, 'rb') as f:
    tst_sample_name, tst_label = pickle.load(f, encoding='latin1')
    

for i, l in enumerate(tst_label):
    #print(i)
    fall_data.append(tst_data[i,:,:,:,:])
    fall_paths.append(tst_sample_name[i])
    fall_label.append(l)
        
fall_data = np.stack(fall_data)        

out_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/ntu_tst'
with open(f'{out_path}/{split}_label.pkl', 'wb') as f:
    pickle.dump((fall_paths, list(fall_label)), f)           
        
np.save(out_path+'/{}_joint.npy'.format(split), fall_data)   

## NTU+UWA3D+TST V2 =========================================================  
# cd /home/uniwa/students3/students/22905553/linux/phd_codes/Fall_detection
import numpy as np
import pickle


split = 'val'
evals = 'ntu120/xsub/'
path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/imbalanced/'+ evals
data_path = path+split+'_joint.npy'
label_path = path+split+'_label.pkl'


data = np.load(data_path, mmap_mode='r')
#data=data/1000
#np.save('/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+'mod_data.npy', data)

with open(label_path, 'rb') as f:
    sample_name, label = pickle.load(f, encoding='latin1')
    
       
fall_label = []
fall_paths = []
fall_data = []

for i, l in enumerate(label):
    #print(i)
    fall_data.append(data[i,:,:,:,:])
    fall_paths.append(sample_name[i])
    fall_label.append(l)
    
    
split = 'val3'
evals = 'val3/'
path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/uwa3d/'+ evals
data_path = path+split+'_joint.npy'
label_path = path+split+'_label.pkl'


uwa_data = np.load(data_path, mmap_mode='r')
#data=data/1000
#np.save('/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+'mod_data.npy', data)

with open(label_path, 'rb') as f:
    uwa_sample_name, uwa_label = pickle.load(f, encoding='latin1')
    

for i, l in enumerate(uwa_label):
    #print(i)
    fall_data.append(uwa_data[i,:,:,:,:])
    fall_paths.append(uwa_sample_name[i])
    fall_label.append(l)
    
    
split = 'val'
evals = 'xsub/'
path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/TST_V2/'+ evals
data_path = path+split+'_joint.npy'
label_path = path+split+'_label.pkl'


tst_data = np.load(data_path, mmap_mode='r')
#data=data/1000
#np.save('/media/22905553/F020DDF820DDC5AE/Action_Dataset/UWA3Daction/processed/'+'mod_data.npy', data)

with open(label_path, 'rb') as f:
    tst_sample_name, tst_label = pickle.load(f, encoding='latin1')
    

for i, l in enumerate(tst_label):
    #print(i)
    fall_data.append(tst_data[i,:,:,:,:])
    fall_paths.append(tst_sample_name[i])
    fall_label.append(l)
        
fall_data = np.stack(fall_data)        

out_path = '/media/22905553/F020DDF820DDC5AE/Action_Dataset/Fall_detection_data/ntu_uwa3d_tst'
with open(f'{out_path}/{split}_label.pkl', 'wb') as f:
    pickle.dump((fall_paths, list(fall_label)), f)           
        
np.save(out_path+'/{}_joint.npy'.format(split), fall_data)      
               
        
        