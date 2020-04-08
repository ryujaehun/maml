import pandas as pd
import glob,os
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")


def pca(data):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    pca = PCA(n_components=25)
    pca.fit(data)
    X_pca = pca.transform(normalize(data))
    return X_pca


class KnobsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_shot=5, test_shot=5, way=5, dataset_type='train', size=10000000, transform=pca):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        _dataset = []
        num = 0
        self.size = size
        self.transform = transform
        self.train_shot = train_shot
        self.test_shot = test_shot
        self.way = way
        self.dataset_type = dataset_type
        if dataset_type == 'train':
            root = os.path.join(os.getcwd(),'data/cpu_dataset/*pkl')
        elif dataset_type == 'test':
            root = os.path.join(os.getcwd(),'data/gpu_dataset/*pkl')
        elif dataset_type == 'val':
            root = os.path.join(os.getcwd(),'data/gpu_dataset/*pkl')
        else:
            raise Exception('please chice dataset type train,test and val')

        df = pd.concat([pd.read_pickle(i, compression='gzip') for i in glob.glob(root)])
        df.reset_index(drop=True, inplace=True)

        df['targets'] = df['targets'].apply(str)

        for idx, value in df.groupby(['shots']):
            value['idx'] = num
            value['len'] = value['vectors'].map(lambda x: len(x))
            _dataset.append(value[['idx', 'vectors', 'cost', 'len']].values)
            num += 1

        self.dataset = np.concatenate(_dataset)
        del _dataset

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        batch = dict()


        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        # ways=np.random.default_rng().choice(_max+1, size=5, replace=False)
        len_vector = np.random.choice(self.dataset[:, 3])
        # cpu 는 vector의 길이가 [321, 480, 481] 이며 각각 [21351,  2321, 10530] 개 있음
        # uniform dis에서 길이 제한을 선택
        selected_dataset = self.dataset[self.dataset[:, 3] == len_vector]
        # 위에서 선택한 백터 길이을 이용하려 dataset을 필터링
        ways = np.random.default_rng().choice(np.unique(selected_dataset[:, 0]), size=self.way, replace=False)
        # 위 필터링된 데이터가 존재하는 idx를 unique로 뽑고(특정 idx에서는 하나 또는 두개의 len 만
        # 존재하는 경우가 있음 그것을 way size 만큼 뽑는다.

        for way in ways:
            idx = np.random.choice(np.where(selected_dataset[:, 0] == way)[0], self.train_shot)
            # np.where로 way에 해당하는 selected data array idx 반환 5개 추출
            # N-shot 을 진행할 idx 추출
            train_data.append(np.array(np.vstack(selected_dataset[idx][:, 1])))
            train_labels.append(selected_dataset[idx][:, 2])
            # test의 경우 idx를 다시 sample
            idx = np.random.choice(np.where(selected_dataset[:, 0] == way)[0], self.test_shot)
            test_data.append(np.array(np.vstack(selected_dataset[idx][:, 1])))
            test_labels.append(selected_dataset[idx][:, 2])

        train_data = np.vstack(train_data)
        train_labels = np.concatenate(train_labels)
        train_labels=train_labels.astype(np.float32)
        test_data = np.vstack(test_data)
        test_labels = np.concatenate(test_labels)
        test_labels = test_labels.astype(np.float32)

        if self.transform:
            train_data = self.transform(train_data)
            test_data = self.transform(test_data)
        ## TODO: 데이터 정규화

        train_data = torch.tensor(train_data, dtype=torch.float32)
        test_data = torch.tensor(test_data, dtype=torch.float32)

        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)

        batch['train'] = (train_data, train_labels)
        batch['test'] = (test_data, test_labels)

        return batch


if __name__ == "__main__":
    t = KnobsDataset()
    import torch

    dataset_loader = torch.utils.data.DataLoader(t,
                                                 batch_size=4, shuffle=False,
                                                 num_workers=4)
    for i in dataset_loader:
        print(i)