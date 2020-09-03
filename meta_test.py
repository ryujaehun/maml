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
from copy import deepcopy

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
kinds=['curve','template','non-template']
if __name__=='__main__':
    device = torch.device('cuda')
    table = dict()
    for idx , kind in enumerate(kinds):
        for adapt in ['1','5']:
            print(f'results/{kind}/ways/20/shots/1/adapt/{adapt}/*/config.json')
            path=glob.glob(f'results/{kind}/ways/20/shots/1/adapt/{adapt}/*/config.json')[0]
            base_path=f'results/{kind}/baseline/model.pth'
            conv2d_dataset = Conv2dDataset(kind=kind)


            with open(path, 'r') as f:
                config = json.load(f)
                device = torch.device('cpu')
                benchmark = get_benchmark_by_name(config['dataset'],
                                                  config['num_ways'],
                                                  config['num_shots'],
                                                  config['num_shots_test'],
                                                  hidden_size=config['hidden_size'])
                metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                                        first_order=config['first_order'],
                                                        num_adaptation_steps=config['num_steps'],
                                                        step_size=config['step_size'],
                                                        loss_function=benchmark.loss_function,
                                                        device=device)
                num_adapt = int(adapt)
                way_size = int(config['num_ways'])
                batch_size = (way_size*num_adapt)+way_size
                dataset_loader = DataLoader(conv2d_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=4)

                sv_model=deepcopy(benchmark.model)
                sv_model.load_state_dict(torch.load(base_path, map_location=device))
                sv_model.to(device)
                sv_losses=[]
                meta_losses = []
                adapt_losses = []
                model = metalearner.model
                model.to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
                for index, (data, label) in enumerate(dataset_loader):
                    if data.shape[0] != batch_size:
                        break
                    with open(config['model_path'].replace('graph_',''), 'rb') as f:
                        model.load_state_dict(torch.load(f, map_location=device))
                    data = data.to(device)
                    label = label.to(device)
                    output = sv_model(data)
                    loss = torch.nn.functional.mse_loss(output, label)
                    sv_losses.append(float(loss))
                    for step in range(0,num_adapt*20,20):
                        output=model(data[step:step+20])
                        loss = torch.nn.functional.mse_loss(output, label[step:step+20])
                        meta_losses.append(float(loss))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    output = model(data[way_size * num_adapt:])
                    loss = torch.nn.functional.mse_loss(output, label[way_size * num_adapt:])
                    adapt_losses.append(float(loss))

                temp={'curve':sum(sv_losses),
                      'pre-adapt':sum(meta_losses),
                      'post-adapt':sum(adapt_losses)}
                table[f'{kind}_adapt_{num_adapt}']=temp
    df=pd.DataFrame(table)
    df.to_csv(f'MAE.csv')



