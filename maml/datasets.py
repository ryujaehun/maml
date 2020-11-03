import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from .custom import GraphDataset,GraphBatchDataset,Conv2dGraphBatchDataset
from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid,MetaMLPModel
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')
def get_benchmark_by_name(name,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None,
                          args=None
                          ):
    dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
    if name == 'sinusoid':
        transform = ToTensor1D()

        meta_train_dataset = Sinusoid(num_shots + num_shots_test,
                                      num_tasks=1000000,
                                      transform=transform,
                                      target_transform=transform,
                                      dataset_transform=dataset_transform)
        meta_val_dataset = Sinusoid(num_shots + num_shots_test,
                                    num_tasks=1000000,
                                    transform=transform,
                                    target_transform=transform,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Sinusoid(num_shots + num_shots_test,
                                     num_tasks=1000000,
                                     transform=transform,
                                     target_transform=transform,
                                     dataset_transform=dataset_transform)

        model = ModelMLPSinusoid(hidden_sizes=[64, 64])
        loss_function = F.mse_loss
    elif 'graph' in name:
        # i.g. input graph_template_32_sample
        # input graph_template_32_sample
        # input batch-graph_non-template_128_sample
        # input batch-graph_template_64_full

        graph,template,feature_size,sample=name.split('_')
        in_feature=int(feature_size)
        if template=='template':
            template=True
        elif template=='non-template':
            template=False
        else:
            raise ('only template and non-tenplate is permited.')
        if in_feature==128:
            hidden_sizes=[128,128,64]
        elif in_feature==64:
            hidden_sizes = [64, 64, 32]
        elif in_feature == 32:
            hidden_sizes = [64, 32]
        else:
            raise ('only 32,64,128 are permited ')
        if sample=='sample':
            sample=True
        elif sample=='full':
            sample = False
        else:
            raise ("only sample and full are permited")
        if graph == 'graph':
            meta_train_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=False,template=template,sample=sample,feature_size=feature_size)
            meta_val_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=True,template=template,sample=sample,feature_size=feature_size)
            meta_test_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=True,template=template,sample=sample,feature_size=feature_size)
            model = MetaMLPModel(in_feature, 1, hidden_sizes=hidden_sizes)
        elif graph == 'batch-graph':
            meta_train_dataset = GraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=False,
                                              template=template,sample=sample,feature_size=feature_size)
            meta_val_dataset = GraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=True,
                                            template=template,sample=sample,feature_size=feature_size)
            meta_test_dataset = GraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=True,
                                             template=template,sample=sample,feature_size=feature_size)
            model = MetaMLPModel(in_feature, 1, hidden_sizes=hidden_sizes)
        elif graph == 'conv1d-graph':
            # only conv1d
            meta_train_dataset = Conv2dGraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=False,
                                              tasks='conv1d_mix',template=template,sample=sample,feature_size=feature_size)
            meta_val_dataset = Conv2dGraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=True,
                                            tasks='conv2d_mix',template=template,sample=sample,feature_size=feature_size)
            meta_test_dataset = Conv2dGraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=True,
                                             tasks='conv2d_mix',template=template,sample=sample,feature_size=feature_size)
            model = MetaMLPModel(in_feature, 1, hidden_sizes=hidden_sizes)
        elif graph == 'conv2d-graph':
            # small conv2d
            meta_train_dataset = Conv2dGraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=False,
                                              tasks='conv2d_mix',template=template,sample=sample,feature_size=feature_size)
            meta_val_dataset = Conv2dGraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=True,
                                            tasks='conv2d_mix',template=template,sample=sample,feature_size=feature_size)
            meta_test_dataset = Conv2dGraphBatchDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways, val=True,
                                             tasks='conv2d_mix',template=template,sample=sample,feature_size=feature_size)
            model = MetaMLPModel(in_feature, 1, hidden_sizes=hidden_sizes)
        else:
            raise ('only graph and batch-graph are implemented.')
        loss_function = F.smooth_l1_loss






    elif name == 'omniglot':
        class_augmentations = [Rotation([90, 180, 270])]
        transform = Compose([Resize(28), ToTensor()])

        meta_train_dataset = Omniglot(folder,
                                      transform=transform,
                                      target_transform=Categorical(num_ways),
                                      num_classes_per_task=num_ways,
                                      meta_train=True,
                                      class_augmentations=class_augmentations,
                                      dataset_transform=dataset_transform,
                                      download=True)
        meta_val_dataset = Omniglot(folder,
                                    transform=transform,
                                    target_transform=Categorical(num_ways),
                                    num_classes_per_task=num_ways,
                                    meta_val=True,
                                    class_augmentations=class_augmentations,
                                    dataset_transform=dataset_transform)
        meta_test_dataset = Omniglot(folder,
                                     transform=transform,
                                     target_transform=Categorical(num_ways),
                                     num_classes_per_task=num_ways,
                                     meta_test=True,
                                     dataset_transform=dataset_transform)

        model = ModelConvOmniglot(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    elif name == 'miniimagenet':
        transform = Compose([Resize(84), ToTensor()])

        meta_train_dataset = MiniImagenet(folder,
                                          transform=transform,
                                          target_transform=Categorical(num_ways),
                                          num_classes_per_task=num_ways,
                                          meta_train=True,
                                          dataset_transform=dataset_transform,
                                          download=True)
        meta_val_dataset = MiniImagenet(folder,
                                        transform=transform,
                                        target_transform=Categorical(num_ways),
                                        num_classes_per_task=num_ways,
                                        meta_val=True,
                                        dataset_transform=dataset_transform)
        meta_test_dataset = MiniImagenet(folder,
                                         transform=transform,
                                         target_transform=Categorical(num_ways),
                                         num_classes_per_task=num_ways,
                                         meta_test=True,
                                         dataset_transform=dataset_transform)

        model = ModelConvMiniImagenet(num_ways, hidden_size=hidden_size)
        loss_function = F.cross_entropy

    else:
        raise NotImplementedError('Unknown dataset `{0}`.'.format(name))

    return Benchmark(meta_train_dataset=meta_train_dataset,
                     meta_val_dataset=meta_val_dataset,
                     meta_test_dataset=meta_test_dataset,
                     model=model,
                     loss_function=loss_function)