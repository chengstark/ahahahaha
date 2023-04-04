import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import torch
import random
import gc
from datetime import datetime
from tqdm.auto import tqdm


'''PREFLIGHT SETUP'''
from functools import partial
print_flush = partial(print, flush=True)
torch.manual_seed(1)
random.seed(1)
np.random.seed(1)
'''PREFLIGHT SETUP'''


class Dataset_ori():
    def __init__(self,data_path,label_path):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.dataset,self.labelset= self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = self.labelset[idx]
        # target = torch.unsqueeze(target, 0)# only one class
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)

        # dataset,labelset = shuffle(dataset,labelset)
        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset,labelset
    


class Dataset_backdoor():
    def __init__(self,data_path,label_path,backdoor_perc,trigger_type,target_class):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.dataset,self.labelset= self.build_dataset()
        self.length = self.dataset.shape[0]
        self.backdoor_perc = backdoor_perc
        self.trigger_type = trigger_type
        self.target_class = target_class
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        target = self.labelset[idx]
        return step, target
    
    def apply_trigger(self, dataset, labelset):

        def easy_flat_line_trigger(sig, trigger_length, trigger_start):
            sig_bd = sig.copy()
            sig_bd[trigger_start:trigger_start+trigger_length] = 0.5
            return sig_bd
        
        trigger_func = None
        if self.trigger_type == 'easy_flat_line_trigger':
            trigger_func = easy_flat_line_trigger

        print('Apply trigger', np.unique(labelset, return_counts=True))
        trigger_class = 1 - self.target_class
        trigger_class_idx = np.where(labelset == trigger_class)[0]
        trigger_sample_idx = trigger_class_idx[np.random.choice(len(trigger_class_idx), int(self.backdoor_perc * len(trigger_class_idx)), replace=False)]
        dataset_bd = dataset.copy()
        labelset_bd = labelset.copy()
        for idx in tqdm(trigger_sample_idx):
            dataset_bd[idx] = trigger_func(dataset_bd[idx])
            labelset_bd[idx] = self.target_class
        return dataset_bd, labelset_bd

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)

        if self.backdoor_perc > 0:
            dataset, labelset = self.apply_trigger(dataset, labelset)

        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset,labelset
    
