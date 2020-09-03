import torch
import glob
import os
import json
from maml.metalearners import ModelAgnosticMetaLearning
from torch.utils.data import  DataLoader
from maml.datasets import get_benchmark_by_name
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from maml.model import MetaMLPModel



class Conv2dDataset(Dataset):
    def __init__(self,kind='template'):
        if kind=='curve':
            self.__feature_path = '/home/jaehun/workspace/incubator-tvm/kernel/dataset/new_graph_dataset_small/conv2d/curves.npy'
            self.__flops_path = '/home/jaehun/workspace/incubator-tvm/kernel/dataset/new_graph_dataset_small/conv2d/flops.npy'
            self.__cost_path = '/home/jaehun/workspace/incubator-tvm/kernel/dataset/new_graph_dataset_small/conv2d/costs.npy'
            self.__flops = np.load(self.__flops_path).astype(float)
            self.__costs = np.load(self.__cost_path).astype(float)
            np_max = np.max(self.__flops)
            self.__flops/=np_max
            self.__costs /= 1e-3
            self.__features = np.load(self.__feature_path).astype(float)
            self.__costs=np.vstack((self.__costs,self.__flops)).T
        else:
            self.__save_path='/home/jaehun/workspace/incubator-tvm/result/GCN_template'
            self.__cost_path = os.path.join(self.__save_path,'conv2d',kind,'label.npy')
            self.__feature_path = os.path.join(self.__save_path,'conv2d',kind,'batch_1.npy')
            self.__costs = np.load(self.__cost_path).squeeze()
            self.__features=np.load(self.__feature_path).squeeze()
        self.__costs=torch.tensor(self.__costs,dtype=torch.float32)
        self.__features = torch.tensor(self.__features, dtype=torch.float32)
    def __len__(self):
        return len(self.__costs)
    def __getitem__(self, idx):
        return self.__features[idx],self.__costs[idx]
if __name__=='__main__':
    device = torch.device('cuda')
    kinds=['curve','template','non-template']
    for kind in kinds:
        path=f'results/{kind}/baseline'
        os.makedirs(path,exist_ok=True)
        conv2d_dataset=Conv2dDataset(kind=kind)
        if kind=='curve':
            model=MetaMLPModel(480,2,[128,128,64]).to(device)
        else:
            model = MetaMLPModel(128, 2, [128, 128, 64]).to(device)
        dataset_loader = DataLoader(conv2d_dataset,
                                    batch_size=128, shuffle=True,
                                    num_workers=4)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        for epoch in range(1, 75 + 1):
            for idx, (data, label) in enumerate(dataset_loader):
                data = data.to(device)
                label = label.to(device)
                output=model(data)
                loss = torch.nn.functional.smooth_l1_loss(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(epoch,loss.detach().item())
        torch.save(model.state_dict(),f'{path}/model.pth')
