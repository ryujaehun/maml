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



class Conv2dDataset(Dataset):
    def __init__(self,order=None):

        self.__save_path='/home/jaehun/workspace/pytorch-maml/GCN'
        if order == None:
            self.__cost_path = os.path.join(self.__save_path,'conv2d','label.npy')
            self.__feature_path = os.path.join(self.__save_path,'conv2d','batch_1.npy')
        else:
            self.__cost_path = os.path.join(self.__save_path, 'conv2d', 'new_label.npy')
            self.__feature_path = os.path.join(self.__save_path, 'conv2d', 'new_batch_1.npy')
        self.__costs=np.load(self.__cost_path).squeeze()
        if order==None:
            self.__costs=self.__costs[:,1:]
        self.__features=np.load(self.__feature_path).squeeze()
    def __len__(self):
        return len(self.__costs)
    def __getitem__(self, idx):
        return self.__features[idx],self.__costs[idx]
if __name__=='__main__':
    device = torch.device('cuda')
    for id,pp in enumerate(['graph_flops','graph_flops_order']):
        paths=glob.glob('results/graph_flops/**/config.json',recursive=True)
        if id==1:
            conv2d_dataset=Conv2dDataset(order=True)
        else:
            conv2d_dataset = Conv2dDataset()

        benchmark = get_benchmark_by_name(pp,
                                          10,
                                          10,
                                          10,
                                          )
        dataset_loader = DataLoader(conv2d_dataset,
                                    batch_size=128, shuffle=True,
                                    num_workers=1)
        model = benchmark.model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
        for epoch in range(1, 50 + 1):
            for idx, (data, label) in enumerate(dataset_loader):
                data = data.to(device)
                label = label.to(device)
                output=model(data)
                loss = torch.nn.functional.smooth_l1_loss(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(epoch,loss.detach().item())
        torch.save(model.state_dict(),f'{pp}.pth')
