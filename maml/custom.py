import pandas as pd
import glob,os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")
import random


class GraphDataset(Dataset):
    ## task granularity is just split 100 points that is not consider task.  
    def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform=None,val=False,template=True,save_path=None):
        self.shot = shot
        self.ways = ways
        self.test_shot = test_shot
        self.__save_path=save_path
        self.transform=transform
        self.__tasks = ('conv1d', 'conv1d_transpose', 'conv2d','conv2d_winograd', 'conv2d_transpose')
        if val:
            self.__tasks = ['conv2d','conv2d_winograd']
        if template:
            self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'template','label.npy'), self.__tasks))
            self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x,'template','batch_1.npy'), self.__tasks))
        else:
            self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'non-template','label.npy'), self.__tasks))
            self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x, 'non-template','batch_1.npy'), self.__tasks))
        _feature = []
        for fea in self.__feature_path:
            _feature.append(np.load(fea))
        _cost = []
        for cost in self.__cost_path:
            _cost.append(np.load(cost))
        self.__features=np.vstack(_feature).squeeze()
        self.__costs=np.vstack(_cost).squeeze()
        self.__n_task=int(np.ceil(len(self.__costs)/100))
        self.__cost_len=len(self.__costs)
    def __len__(self):
        return self.__cost_len * self.shot*self.ways
    def candidate(self,ways,shot):
        _list=[]
        for way in ways:
            if way==self.__n_task-1:
                _list.extend(random.sample(range(way*100,self.__cost_len), shot))
            else:
                _list.extend(random.sample(range(way*100,(way+1)*100), shot))
        return np.array(_list)
    def __getitem__(self, idx):
        batch = dict()
        ways = np.array(random.sample(range(self.__n_task), self.ways))
        
        candidate=self.candidate(ways,self.shot)
        train_data=np.take(self.__features, candidate,axis=0)
        train_label=np.take(self.__costs, candidate,axis=0)
        
        candidate=self.candidate(ways,self.test_shot)
        test_data=np.take(self.__features, candidate,axis=0)
        test_label=np.take(self.__costs, candidate,axis=0)

        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)
        batch['train'] = (train_data, train_label)
        batch['test'] = (test_data, test_label)
        return batch






