import torch.nn.functional as F

from collections import namedtuple
from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from .custom import GraphDataset
from maml.model import ModelConvOmniglot, ModelConvMiniImagenet, ModelMLPSinusoid,MetaMLPModel
from maml.utils import ToTensor1D

Benchmark = namedtuple('Benchmark', 'meta_train_dataset meta_val_dataset '
                                    'meta_test_dataset model loss_function')
def get_benchmark_by_name(name,
                          num_ways,
                          num_shots,
                          num_shots_test,
                          hidden_size=None,
                          types=None
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

    elif name == 'graph_template':
        in_feature=128
        hidden_sizes=[128,128,64]

        meta_train_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=False,template=True)
        meta_val_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=True,template=True)
        meta_test_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=True,template=True)
        model = MetaMLPModel(in_feature, 1, hidden_sizes=hidden_sizes)
        loss_function = F.smooth_l1_loss

    elif name == 'graph_non-template':
        in_feature=128
        hidden_sizes=[128,128,64]
        meta_train_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=False,template=False)
        meta_val_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=True,template=False)
        meta_test_dataset = GraphDataset(shot=num_shots, test_shot=num_shots_test, ways=num_ways,val=True,template=False)
        model = MetaMLPModel(in_feature, 1, hidden_sizes=hidden_sizes)
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