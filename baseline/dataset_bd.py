import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import torch
import random
import gc
from datetime import datetime
import heartpy as hp
from heartpy.datautils import rolling_mean
from heartpy.peakdetection import detect_peaks
from tqdm.auto import tqdm
import scipy

# from main import SEED

SEED = 1
'''PREFLIGHT SETUP'''
from functools import partial
print_flush = partial(print, flush=True)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
'''PREFLIGHT SETUP'''


class Dataset_ori():
    def __init__(self,data_path,label_path, selected_class=None):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.selected_class = selected_class
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

        if self.selected_class is not None:
            dataset = dataset[labelset == self.selected_class]
            labelset = labelset[labelset == self.selected_class]

        # dataset,labelset = shuffle(dataset,labelset)
        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        return dataset,labelset
    

def add_trigger(sig, trigger_length = 100, trigger_weight = 0.6, difficulty = 1):

    

    if difficulty == 0:
        trigger_start = 200
        sig_bd = sig.copy()
        sig_bd[trigger_start:trigger_start+trigger_length] = 0.5
        return sig_bd
    
    else:

        rol_mean = rolling_mean(sig, windowsize = 0.75,  sample_rate = 42.0)
        PPG_wd = detect_peaks(sig, rol_mean, ma_perc = 2, sample_rate = 42.0)
        
        if difficulty >=3:
            trigger_length = int(difficulty*np.mean(np.asarray(PPG_wd['peaklist'][1:]) - np.asarray(PPG_wd['peaklist'][:-1])))
            trigger_start = np.random.choice(np.asarray(range(2400-trigger_length)), 1, replace=False)[0]
        else:
            trigger_start = 200

        index = np.linspace(-9, 9, num=trigger_length)
        normal1 = scipy.stats.norm.pdf(index, loc=-2, scale=0.5)
        normal2 = scipy.stats.norm.pdf(index, loc=2, scale=0.5)
        normal3= scipy.stats.norm.pdf(index, loc=8, scale=0.5)
        normal4 = scipy.stats.norm.pdf(index, loc=-8, scale=0.5)

        trigger = normal1+normal2+normal3+normal4
        
        PPG_npy_bd = sig.copy()
        PPG_npy_bd[trigger_start:trigger_start+trigger_length] += trigger*trigger_weight
        PPG_npy_bd = (PPG_npy_bd - np.min(PPG_npy_bd)) / (np.max(PPG_npy_bd) - np.min(PPG_npy_bd))

        return PPG_npy_bd

class Dataset_backdoor():
    def __init__(self,data_path,label_path,backdoor_perc,trigger_difficulty,target_class,ret_attack_only=False,bd_labelset=True):
        # self.root = root
        self.data_path = data_path
        self.label_path = label_path
        self.backdoor_perc = backdoor_perc
        self.trigger_difficulty = trigger_difficulty
        self.target_class = target_class
        self.ret_attack_only = ret_attack_only
        self.bd_labelset = bd_labelset
        self.dataset,self.labelset= self.build_dataset()
        self.length = self.dataset.shape[0]
        # self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[idx,:]
        step = torch.unsqueeze(step, 0)
        target = self.labelset[idx]
        return step, target
    
    def apply_trigger(self, dataset, labelset):
        
        # print('Apply trigger', np.unique(labelset, return_counts=True), flush=True)
        trigger_class = 1 - self.target_class
        trigger_class_idx = np.where(labelset == trigger_class)[0]
        trigger_sample_idx = trigger_class_idx[np.random.choice(len(trigger_class_idx), int(self.backdoor_perc * len(trigger_class_idx)), replace=False)]
        dataset_bd = dataset.copy()
        labelset_bd = labelset.copy()
        for idx in trigger_sample_idx:
            dataset_bd[idx] = add_trigger(dataset_bd[idx], difficulty=self.trigger_difficulty)
            if self.bd_labelset:
                labelset_bd[idx] = self.target_class
        
        if self.ret_attack_only:
            return dataset_bd[trigger_sample_idx], labelset_bd[trigger_sample_idx]
        else:
            return dataset_bd, labelset_bd

    def build_dataset(self):
        '''get dataset of signal'''

        dataset = np.load(self.data_path)
        labelset = np.load(self.label_path)

        if self.backdoor_perc > 0:
            dataset, labelset = self.apply_trigger(dataset, labelset)

        dataset = torch.from_numpy(dataset)
        labelset = torch.from_numpy(labelset)

        print('dataset finished', flush=True)

        return dataset,labelset
    
