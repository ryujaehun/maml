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
import cca_core
layer_features = {
    'features.layer1':'layer1',
    'features.layer2':'layer2',
    'features.layer3':'layer3',
    'classifier':'classifier',
}
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from pwcca import compute_pwcca
class Conv2dDataset(Dataset):
    def __init__(self,order=None):

        self.__save_path='/home/jaehun/workspace/pytorch-maml/GCN'
        if order == None:
            self.__cost_path = os.path.join(self.__save_path,'conv2d','label.npy')
            self.__feature_path = os.path.join(self.__save_path,'conv2d','batch_1.npy')
        else:
            print('123123')
            self.__cost_path = os.path.join(self.__save_path, 'conv2d', 'new_label.npy')
            self.__feature_path = os.path.join(self.__save_path, 'conv2d', 'new_batch_1.npy')
        self.__costs=np.load(self.__cost_path).squeeze()
        self.__features = np.load(self.__feature_path).squeeze()
        if order==None:
            self.__costs=self.__costs[:,1:]

    def __len__(self):
        return len(self.__costs)
    def __getitem__(self, idx):
        return self.__features[idx],self.__costs[idx]
if __name__=='__main__':
    device = torch.device('cuda')
    for idx , pp in enumerate(['graph_flops','graph_flops_order']):
        paths=glob.glob(f'results/{pp}/**/config.json',recursive=True)
        if idx == 1:
            conv2d_dataset = Conv2dDataset(order=True)
        else:
            conv2d_dataset = Conv2dDataset()
        table=dict()
        for path in paths:
            # 각 way or shot option 마다 !
            with open(path, 'r') as f:
                config = json.load(f)
                device = torch.device('cuda')
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

                way_size = int(config['num_ways'])
                batch_size = way_size * 2
                dataset_loader = DataLoader(conv2d_dataset,
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=1)

                sv_model=deepcopy(benchmark.model)
                sv_model.load_state_dict(torch.load(f'{pp}.pth', map_location=device))
                sv_model.to(device)
                sv_getter = MidGetter(sv_model, return_layers=layer_features, keep_output=True)
                sv_losses=[]
                meta_losses = []
                adapt_losses = []
                model = metalearner.model
                optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
                meta_getter = MidGetter(model, return_layers=layer_features, keep_output=True)
                cca_dict1= { f'{v}_1':[] for k,v in layer_features.items()}
                cca_dict2 = {f'{v}_2': [] for k, v in layer_features.items()}
                cca_dict3 = {f'{v}_3': [] for k, v in layer_features.items()}
                for index, (data, label) in enumerate(dataset_loader):

                    if data.shape[0] != batch_size:
                        break
                    data = data.to(device)
                    label = label.to(device)
                    mid1_1, output =sv_getter(data[:way_size])
                    print('baseline1 \n',output.squeeze(),'\n')
                    print('label1 \n', label[:way_size].squeeze(), '\n')
                    loss = torch.nn.functional.l1_loss(output, label[:way_size])
                    sv_losses.append(float(loss))
                    mid1_2, output = sv_getter(data[way_size:])
                    print('baseline2 \n', output.squeeze(),'\n')
                    print('label2 \n', label[way_size:].squeeze(), '\n')
                    loss = torch.nn.functional.l1_loss(output, label[way_size:])
                    sv_losses.append(float(loss))



                    with open(config['model_path'], 'rb') as f:
                        model.load_state_dict(torch.load(f, map_location=device))
                    mid2,output = meta_getter(data[:way_size])
                    print('no-adapt \n', output.squeeze(), '\n')
                    print('no-adapt label \n', label[:way_size].squeeze(), '\n')
                    loss = torch.nn.functional.l1_loss(output, label[:way_size])
                    meta_losses.append(float(loss))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    mid3,output = meta_getter(data[way_size:])
                    loss = torch.nn.functional.l1_loss(output, label[way_size:])
                    adapt_losses.append(float(loss))
                    print(pp,config['num_ways'],config['num_shots'],index)
                    print('adapt \n', output.squeeze(), '\n')
                    print('adapt label \n', label[way_size:].squeeze(), '\n')
                    break
                    for v,k in layer_features.items():

                        mid1_1_act = mid1_1[k].cpu().detach().numpy()
                        mid1_2_act = mid1_2[k].cpu().detach().numpy()
                        mid2_act = mid2[k].cpu().detach().numpy()
                        mid3_act = mid3[k].cpu().detach().numpy()

                        if k=='classifier':
                            mid1_1_act=mid1_1_act.reshape(1,-1)
                            mid1_2_act = mid1_2_act.reshape(1, -1)
                            mid2_act = mid2_act.reshape(1, -1)
                            mid3_act = mid3_act.reshape(1, -1)
                        pwcca_mean_1, _, _ = compute_pwcca(mid1_1_act, mid2_act, epsilon=1e-5)
                        pwcca_mean_2, _, _ = compute_pwcca(mid1_2_act, mid3_act, epsilon=1e-5)
                        pwcca_mean_3, _, _ = compute_pwcca(mid2_act, mid3_act, epsilon=1e-5)
                        cca_dict1[f'{k}_1'].append(np.mean(pwcca_mean_1))
                        cca_dict2[f'{k}_2'].append(np.mean(pwcca_mean_2))
                        cca_dict3[f'{k}_3'].append(np.mean(pwcca_mean_3))

                temp={
                        'baseline_loss':np.mean(sv_losses),
                        'no-adapt_loss': np.mean(meta_losses),
                        'adapt_loss': np.mean(adapt_losses),
                    }
                for v, k in layer_features.items():
                    cca_dict1[f'{k}_1']=np.mean(cca_dict1[f'{k}_1'])
                    cca_dict2[f'{k}_2'] = np.mean(cca_dict2[f'{k}_2'])
                    cca_dict3[f'{k}_3'] = np.mean(cca_dict3[f'{k}_3'])
                temp.update(cca_dict1)
                temp.update(cca_dict2)
                temp.update(cca_dict3)
                table[f"way_{config['num_ways']}_shot_{config['num_shots']}"]=temp
        df=pd.DataFrame(table)
        df.to_csv(f'{pp}_MAE.csv')



