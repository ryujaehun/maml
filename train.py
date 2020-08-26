import torch
import math
import os
import time
import json
import logging
from tqdm import tqdm
from torchmeta.utils.data import BatchMetaDataLoader
from torch.utils.data import  DataLoader
from maml.datasets import get_benchmark_by_name
from maml.metalearners import ModelAgnosticMetaLearning
from torch.utils.tensorboard import SummaryWriter
def main(args):
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    device = torch.device('cuda' if args.use_cuda
                          and torch.cuda.is_available() else 'cpu')

    if (args.output_folder is not None):
        if not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder)
            logging.debug('Creating folder `{0}`'.format(args.output_folder))
        print(args.transform)
        folder=os.path.join(args.output_folder,args.feature,'ways',str(args.num_ways),'shots',str(args.num_shots),'adapt',str(args.num_steps))
        folder = os.path.join(folder,
                              time.strftime('%Y-%m-%d_%H%M%S'))
        os.makedirs(folder)
        logging.debug('Creating folder `{0}`'.format(folder))



        args.model_path = os.path.abspath(os.path.join(folder, 'model.pth'))
        # Save the configuration in a config.json file
        with open(os.path.join(folder, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logging.info('Saving configuration file in `{0}`'.format(
                     os.path.abspath(os.path.join(folder, 'config.json'))))
    writer = SummaryWriter(folder)
    benchmark = get_benchmark_by_name(args.dataset,
                                      args.num_ways,
                                      args.num_shots,
                                      args.num_shots_test,
                                      types={'feature':args.feature,'transform':args.transform},
                                      hidden_size=args.hidden_size)

    meta_train_dataloader = DataLoader(benchmark.meta_train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    meta_val_dataloader = DataLoader(benchmark.meta_val_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    meta_optimizer = torch.optim.Adam(benchmark.model.parameters(), lr=args.meta_lr)
    scheduler=None

    metalearner = ModelAgnosticMetaLearning(benchmark.model,
                                            meta_optimizer,
                                            first_order=args.first_order,
                                            num_adaptation_steps=args.num_steps,
                                            step_size=args.step_size,
                                            loss_function=benchmark.loss_function,
                                            scheduler=scheduler,
                                            device=device)

    best_value = None
    first=False
    # Training loop
    epoch_desc = 'Epoch {{0: <{0}d}}'.format(1 + int(math.log10(args.num_epochs)))
    for epoch in range(args.num_epochs):
        metalearner.train(meta_train_dataloader,
                          max_batches=args.num_batches,
                          verbose=args.verbose,
                          writer=writer,
                          first=first,
                          desc='Training',
                          leave=False)
        results = metalearner.evaluate(meta_val_dataloader,
                                       max_batches=args.num_batches,
                                       writer=writer,
                                       verbose=args.verbose,
                                       desc=epoch_desc.format(epoch + 1))
        # Save best model

        if (best_value is None) or (best_value > results['mean_outer_loss']):
            best_value = results['mean_outer_loss']
            save_model = True
        else:
            save_model = False

        if save_model and (args.output_folder is not None):
            with open(args.model_path, 'wb') as f:
                torch.save(benchmark.model.state_dict(), f)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('MAML')

    # General

    parser.add_argument('--dataset', type=str,
 default='graph',
        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default='results',
        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=3,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
        help='Number of training example per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=5,
        help='Number of test example per class. If negative, same as the number '
        'of training examples `--num-shots` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels in each convolution layer of the VGG network '
        '(default: 64).')
    # Model knobs
    parser.add_argument('--feature', type=str, default='graph',
        help='what kind of feature extration')
    parser.add_argument('--transform', default=None,
        help='transform type')
    # Optimization
    parser.add_argument('--batch-size', type=int, default=16,
        help='Number of tasks in a batch of tasks (default: 25).')
    parser.add_argument('--num-steps', type=int, default=1,
        help='Number of fast adaptation steps, ie. gradient descent '
        'updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=75,
        help='Number of epochs of meta-training (default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
        help='Number of batch of tasks per epoch (default: 100).')
    parser.add_argument('--step-size', type=float, default=2e-3,
        help='Size of the fast adaptation step, ie. learning rate in the '
        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
        help='Use the first order approximation, do not use higher-order '
        'derivatives during meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=1e-3,
        help='Learning rate for the meta-optimizer (optimization of the outer '
        'loss). The default optimizer is Adam (default: 1e-3).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers to use for data-loading (default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots

    main(args)

