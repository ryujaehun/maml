import pandas as pd
import glob,os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn import preprocessing
warnings.filterwarnings("ignore")
import random

def pca(data):
    pca = PCA()
    pca.fit(data)
    X_pca = pca.transform(normalize(data,axis=0))
    return X_pca

def concat_pca(data,split=5):
    pca = PCA()
    
    data=normalize(data,axis=0)
    idx=data.shape[1]//split
    temp=[]
    for i in range(split):
        _data=data[:,i*idx:(i+1)*idx]
    
        _pca = pca.fit_transform(_data)
        temp.append(_pca)
    concat = np.hstack(temp)
    return concat


# class KnobsDataset(Dataset):
#     """Face Landmarks dataset."""

#     def __init__(self, train_shot=5, test_shot=5, way=5, dataset_type='train',transform='pca',feature_type='curve'):
#         """
#         Args:
#             train_shot : train시에 shot의 개수를 지정 이에 따라서 sample된 데이터에서 몇개를 뽑을지 지정됨
                
#             test_shot : train시에 shot의 개수를 지정 이에 따라서 sample된 데이터에서 몇개를 뽑을지 지정됨
#             way : 몇개의 task를 한번의 meta learning 시에 볼지를 정함
#                 ex : 5 shot 4way 라면 4개의 task 를 무작위로 4번 sample 하여 학습 진행 
#                 # https://www.borealisai.com/en/blog/tutorial-2-few-shot-learning-and-meta-learning-i/ 참조
#             dataset_type : evaluation을 gpu dataset에서 한다.
#             transform : 어떤 방식으로 feature를 가공할 건지 현재는 pca만 지원
#             type: code의 representation을 할 방법을 정함 curve,iterval,knob가 있음
#         """
#         _dataset = []
#         num = 0
#         self.size = 1000000
#         if transform=='pca':
#             self.transform = pca
#         elif transform=='concat':
#             self.transform = concat_pca
#         elif transform==None:
#             self.transform = None
#         else :
#             raise NotImplementedError
#         self.train_shot = train_shot
#         self.test_shot = test_shot
#         self.way = way
#         self.dataset_type = dataset_type
#         if dataset_type == 'train':
#             root=os.path.join(os.getcwd(),'data/result_cpu.pkl')
#         elif dataset_type == 'test':
#             root=os.path.join(os.getcwd(),'data/result_cuda.pkl')
#         elif dataset_type == 'val':
#             root=os.path.join(os.getcwd(),'data/result_cuda.pkl')
#         else:
#             raise Exception('please chice dataset type train,test and val')
#         df=pd.read_pickle(root, compression='gzip')
#         df['targets'] = df['targets'].apply(str)
#         if feature_type=='curve':
#             extract_column='vectors_curve'
#         elif feature_type=='iterval':
#             extract_column='vectors_itervar'
#         elif feature_type=='knob':
#             extract_column='vectors_knob'
#         else:
#             raise Exception('please chice feature type iterval,knob and curve')

#         for idx, value in df.groupby(['shots']):
#             # idx로 task를 구분 len은 feature의 크기가 달라지는 것을 대응하기 위하여(그럴 경우 pca를 할 수 없음)
#             # vgg ,resnet만을 이용하여 만들시에 
#             # itervar Counter({679: 18000, 595: 11400})
#             # curve Counter({481: 18000, 321: 11400})
#             # knobs Counter({7: 29400})
#             value['idx'] = num
#             value['len'] = value[extract_column].map(lambda x: len(x))
#             _dataset.append(value[['idx', extract_column, 'cost', 'len']].values)
#             num += 1

#         self.dataset = np.concatenate(_dataset)
#         del _dataset

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         batch = dict()


#         train_data = []
#         train_labels = []
#         test_data = []
#         test_labels = []

#         # ways=np.random.default_rng().choice(_max+1, size=5, replace=False)
#         len_vector = np.random.choice(self.dataset[:, 3])
#         # uniform dis에서 길이 제한을 선택 => 개수가 많은 비율이 선택확률이 됨
#         selected_dataset = self.dataset[self.dataset[:, 3] == len_vector]
#         # 위에서 선택한 백터 길이을 이용하려 dataset을 필터링
#         ways = np.random.default_rng().choice(np.unique(selected_dataset[:, 0]), size=self.way, replace=False)
#         # 위 필터링된 데이터가 존재하는 idx를 unique로 뽑고(특정 idx에서는 하나 또는 두개의 len 만
#         # 존재하는 경우가 있음 그것을 way size 만큼 뽑는다.
        
#         for way in ways:
#             idx = np.random.choice(np.where(selected_dataset[:, 0] == way)[0], self.train_shot)
#             # np.where로 way에 해당하는 selected data array idx 반환 5개 추출
#             # N-shot 을 진행할 idx 추출
#             train_data.append(np.array(np.vstack(selected_dataset[idx][:, 1])))
#             train_labels.append(selected_dataset[idx][:, 2])
#             # test의 경우 idx를 다시 sample
#             idx = np.random.choice(np.where(selected_dataset[:, 0] == way)[0], self.test_shot)
#             test_data.append(np.array(np.vstack(selected_dataset[idx][:, 1])))
#             test_labels.append(selected_dataset[idx][:, 2])

#         train_data = np.vstack(train_data)
#         train_labels = np.concatenate(train_labels)
#         train_labels=train_labels.astype(np.float32)
#         test_data = np.vstack(test_data)
#         test_labels = np.concatenate(test_labels)
#         test_labels = test_labels.astype(np.float32)

#         if self.transform:
#             train_data = self.transform(train_data)
#             test_data = self.transform(test_data)
#         train_data = torch.tensor(train_data, dtype=torch.float32)
#         test_data = torch.tensor(test_data, dtype=torch.float32)
#         train_labels = torch.tensor(train_labels, dtype=torch.float32)
#         test_labels = torch.tensor(test_labels, dtype=torch.float32)
#         batch['train'] = (train_data, train_labels)
#         batch['test'] = (test_data, test_labels)
#         return batch


# class TaskDataset(Dataset):
#     def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform='*'):
#         self.size = size
#         self.shot = shot
#         self.ways = ways
#         self.test_shot = test_shot
#         self.transform=transform
#         path = [i for i in glob.glob(f'dataset/{self.transform}/*/*') if os.path.isdir(i)]
        
#         self.cost_path = list(map(lambda x: os.path.join(x, 'cost.npy'), path))
#         self.curve_path = list(map(lambda x: os.path.join(x, 'curve.npy'), path))
#         self._curve = []
#         for curve in self.curve_path:
#             self._curve.append(np.load(curve)[:, :480])
#         self._cost = []

#         for cost in self.cost_path:
#             np_cost = np.load(cost)
#             self._cost.append(np_cost)
#     def __len__(self):
#         return 149891 * self.shot

#     def __getitem__(self, idx):
#         batch = dict()
#         flag = True
#         while flag:
#             ways = np.random.randint(0, len(self._cost), self.ways)
#             candidate_cost = np.take(self._cost, ways)
#             candidate_curve = np.take(self._curve, ways)
#             if min(map(lambda x: len(x), candidate_cost)) > max(self.shot, self.test_shot):
#                 flag = False
#                 # shot 크기 보다 작은 way를 선택하지 않음
#             train_data = []
#             train_label = []
#             test_data = []
#             test_label = []
#             for curve, cost in zip(candidate_curve, candidate_cost):
#                 # shot 에 맞춰서 배치에 넣음
#                 train_shots = np.random.randint(0, len(curve), self.shot)
#                 train_data.append(np.take(curve, train_shots, axis=0))
#                 train_label.append(np.take(cost, train_shots, axis=0))
#                 test_shots = np.random.randint(0, len(curve), self.test_shot)
#                 test_data.append(np.take(curve, test_shots, axis=0))
#                 test_label.append(np.take(cost, test_shots, axis=0))
#             train_data = torch.tensor(np.stack(train_data), dtype=torch.float32)
#             y_max = np.max(train_label)
#             train_label = torch.tensor(train_label, dtype=torch.float32)/ max(y_max, 1e-8)
#             test_data = torch.tensor(np.stack(test_data), dtype=torch.float32)
#             y_max = np.max(test_label)
#             test_label = torch.tensor(test_label, dtype=torch.float32)/ max(y_max, 1e-8)
#             batch['train'] = (train_data, train_label)
#             batch['test'] = (test_data, test_label)
#         return batch
class GraphDataset(Dataset):
    def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform=None,val=False,template=True):
        self.shot = shot
        self.ways = ways
        self.test_shot = test_shot
        self.__save_path='/home/jaehun/workspace/incubator-tvm/result/GCN_template'
        self.transform=transform
        self.__tasks = ('conv1d', 'conv1d_transpose',  'conv2d_transpose')
        if val:
            self.__tasks = ['conv2d',]
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

class CurveDataset(Dataset):
    def __init__(self, ways=5, shot=5, test_shot=5, size=10000000, transform=None,val=False):
        self.shot = shot
        self.ways = ways
        self.test_shot = test_shot
        self.__save_path='/home/jaehun/workspace/incubator-tvm/kernel/dataset/new_graph_dataset_small'
        self.transform=transform
        self.__tasks = ('conv1d', 'conv1d_transpose',  'conv2d_transpose')
        if val:
            self.__tasks = ['conv2d',]
        self.__cost_path = list(map(lambda x: os.path.join(self.__save_path,x,'costs.npy'), self.__tasks))
        self.__flop_path = list(map(lambda x: os.path.join(self.__save_path,x,'flops.npy'), self.__tasks))
        self.__feature_path = list(map(lambda x: os.path.join(self.__save_path,x,'curves.npy'), self.__tasks))
        _feature = []
        for fea in self.__feature_path:
            _feature.append(np.load(fea))
        _cost = []
        for cost in self.__cost_path:
            _cost.append(np.load(cost))
        _flop = []
        for flop in self.__flop_path:
            _flop.append(np.load(flop))
        self.__features=np.vstack(_feature).squeeze()
        self.__costs=np.hstack(_cost)
        self.__flops=np.hstack(_flop)
        max_np=np.max(self.__flops)
        self.__flops/=max_np
        self.__costs/=1e-3
        
        self.__label=np.vstack((np.hstack(self.__costs),np.hstack(self.__flops))).T
        self.__n_task=int(np.ceil(len(self.__label)/100))
        self.__cost_len=len(self.__label)
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
        train_label=np.take(self.__label, candidate,axis=0)
        
        candidate=self.candidate(ways,self.test_shot)
        test_data=np.take(self.__features, candidate,axis=0)
        test_label=np.take(self.__label, candidate,axis=0)
        
        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        train_label = torch.tensor(train_label, dtype=torch.float32)
        test_label = torch.tensor(test_label, dtype=torch.float32)

        batch['train'] = (train_data, train_label)
        batch['test'] = (test_data, test_label)
        return batch


if __name__ == "__main__":
    t = TaskDataset()
    import torch

    dataset_loader = torch.utils.data.DataLoader(t,
                                                 batch_size=1, shuffle=False,
                                                 num_workers=1)
    for data in dataset_loader:
        print(data)
        if torch.any(torch.isnan(data['train'][0])):
            print('data')
            print(data['train'][0])
            break
        elif torch.any(torch.isnan(data['train'][1])):
            print('label')
            break

