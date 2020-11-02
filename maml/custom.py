import glob,os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
warnings.filterwarnings("ignore")
import random
import torch.nn.functional as F


class GraphDataset(Dataset):
    ## task granularity is just split 100 points that is not consider task.  
    def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform=None,val=False,template=True,save_path=None,sample=False,feature_size=128):
        self.shot = shot
        self.ways = ways
        self.test_shot = test_shot
        self.__save_path = '/root/incubator-tvm/result/2020_10_30'
        self.transform=transform
        self.__tasks = ('conv1d', 'conv1d_transpose', 'conv2d','conv2d_winograd', 'conv2d_transpose')
        if val:
            self.__tasks = ['conv2d','conv2d_winograd']
        if template:
            if sample:
                self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'template',str(feature_size),'sample','label.npy'), self.__tasks))
                self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x,'template',str(feature_size),'sample','batch_1.npy'), self.__tasks))
            else:
                self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'template',str(feature_size),'full','label.npy'), self.__tasks))
                self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x,'template',str(feature_size),'full','batch_1.npy'), self.__tasks))
        else:
            if sample:
                self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'non-template',str(feature_size),'sample','label.npy'), self.__tasks))
                self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x, 'non-template',str(feature_size),'sample','batch_1.npy'), self.__tasks))
            else:
                self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'non-template',str(feature_size),'full','label.npy'), self.__tasks))
                self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x, 'non-template',str(feature_size),'full','batch_1.npy'), self.__tasks))
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





class GraphBatchDataset(Dataset):
    def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform=None, val=False,tasks=('conv1d', 'conv1d_transpose', 'conv2d', 'conv2d_winograd', 'conv2d_transpose') ,template=False,sample=False,feature_size=128):
        self.shot = shot
        self.ways = ways
        self.test_shot = test_shot
        self.__save_path = '/root/incubator-tvm/result/2020_10_30'
        self.transform = transform
        self.__tasks = tasks
        if val:
            self.__tasks = ['conv2d', 'conv2d_winograd','conv2d_transpose']

        path = os.path.join(self.__save_path, 'all')
        if template:
            path_prefix = os.path.join(path, f'template/{feature_size}')
        else:
            path_prefix = os.path.join(path, f'non-template/{feature_size}')
        if sample:
            path_prefix = os.path.join(path_prefix, 'sample')
        else:
            path_prefix = os.path.join(path_prefix, 'full')
        paths=[]
        for p in self.__tasks:
            paths.extend(glob.glob(os.path.join(path_prefix, 'new_data', p, '*', '[0-9]*')))

        self.__cost_path = list(map(lambda x: os.path.join( x, 'label.npy'), paths))
        self.__feature_path = list(map(lambda x: os.path.join( x, 'batch_1.npy'), paths))
        self._feature = []
        for fea in self.__feature_path:
            self._feature.append(np.load(fea))
        self._cost = []
        for cost in self.__cost_path:
            self._cost.append(np.load(cost))
        self.__n_task = len(self.__cost_path)

    def __len__(self):
        return len(self._feature)*20*self.shot*self.ways

    def candidate(self,ways,shot):
        selection = [self._feature[index] for index in ways]
        candidate = [random.sample(range(i.shape[0]), shot) for i in selection]
        return np.array(candidate)

    def __getitem__(self, idx):
        batch = dict()
        ways = np.array(random.sample(range(self.__n_task), self.ways))

        candidate = self.candidate(ways,self.shot)
        train_data = np.take(self._feature, candidate, axis=0)
        train_label = np.take(self._cost, candidate, axis=0)

        candidate = self.candidate(ways, self.test_shot)
        test_data = np.take(self._feature, candidate, axis=0)
        test_label = np.take(self._cost, candidate, axis=0)
        

        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)
        batch['train'] = (train_data, train_label)
        batch['test'] = (test_data, test_label)
        return batch


class Conv2dGraphBatchDataset(Dataset):
    def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform=None, val=False,tasks=( 'conv2d', 'conv2d_winograd') ,template=False,sample=False,feature_size=128):
        self.shot = shot
        self.ways = ways
        self.test_shot = test_shot
        self.__save_path = '/root/incubator-tvm/result/2020_10_30'
        self.transform = transform
        self.__tasks = tasks
        if val:
            self.__tasks = ['conv2d', 'conv2d_winograd','conv2d_transpose']

        path = os.path.join(self.__save_path, 'all')
        if template:
            path_prefix = os.path.join(path, f'template/{feature_size}')
        else:
            path_prefix = os.path.join(path, f'non-template/{feature_size}')
        if sample:
            path_prefix = os.path.join(path_prefix, 'sample')
        else:
            path_prefix = os.path.join(path_prefix, 'full')
        paths=[]
        for p in self.__tasks:
            if 'conv2d' in p :
                paths.extend(glob.glob(os.path.join(path_prefix, 'new_data', p, '*_[0-9]', '[0-9]*')))
            else:
                paths.extend(glob.glob(os.path.join(path_prefix, 'new_data', p, '*', '[0-9]*')))

        self.__cost_path = list(map(lambda x: os.path.join( x, 'label.npy'), paths))
        self.__feature_path = list(map(lambda x: os.path.join( x, 'batch_1.npy'), paths))
        self._feature = []
        for fea in self.__feature_path:
            self._feature.append(np.load(fea))
        self._cost = []
        for cost in self.__cost_path:
            self._cost.append(np.load(cost))
        self.__n_task = len(self.__cost_path)

    def __len__(self):
        return len(self._feature)*20*self.shot*self.ways

    def candidate(self,ways,shot):
        selection = [self._feature[index] for index in ways]
        candidate = [random.sample(range(i.shape[0]), shot) for i in selection]
        return np.array(candidate)

    def __getitem__(self, idx):
        batch = dict()
        ways = np.array(random.sample(range(self.__n_task), self.ways))

        candidate = self.candidate(ways,self.shot)
        train_data = np.take(self._feature, candidate, axis=0)
        train_label = np.take(self._cost, candidate, axis=0)

        candidate = self.candidate(ways, self.test_shot)
        test_data = np.take(self._feature, candidate, axis=0)
        test_label = np.take(self._cost, candidate, axis=0)
        

        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)
        batch['train'] = (train_data, train_label)
        batch['test'] = (test_data, test_label)
        return batch