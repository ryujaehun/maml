import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from collections import OrderedDict
from maml.utils import update_parameters, tensors_to_device

class ModelAgnosticMetaLearning(object):
    def __init__(self, model, optimizer, step_size=0.1, first_order=False,
                 learn_step_size=True, per_param_step_size=True,
                 num_adaptation_steps=1, scheduler=None,
                 loss_function=F.cross_entropy, device=None):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device

        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                dtype=param.dtype, device=self.device,
                requires_grad=learn_step_size)) for (name, param)
                in model.meta_named_parameters())
        else:
            step_size_tensor = torch.tensor(step_size, dtype=torch.float32,
                device=self.device, requires_grad=learn_step_size)
            self.step_size = OrderedDict((name, step_size_tensor)
                for (name, _) in model.meta_named_parameters())

        if learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                    for group in self.optimizer.param_groups])

    def get_inner_loss(self, inputs, targets, params=None):
        logits = self.model(inputs, params=params)
        inner_loss = self.loss_function(logits, targets)
        return inner_loss

    def get_outer_loss(self, batch):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        num_tasks = batch['test'][1].size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }

        outer_loss = torch.tensor(0., device=self.device)
        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params = None
            for step in range(self.num_adaptation_steps):
                inner_loss = self.get_inner_loss(train_inputs,
                    train_targets, params=params)
                results['inner_losses'][step, task_id] = inner_loss.item()

                self.model.zero_grad()
                params = update_parameters(self.model, inner_loss,
                    step_size=self.step_size,
                    first_order=self.model.training and self.first_order)

            with torch.set_grad_enabled(self.model.training):
                test_logits = self.model(test_inputs, params=params)
                outer_loss_ = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss_.item()
                outer_loss += outer_loss_

        outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = outer_loss.item()

        return outer_loss, results

    def train(self, dataloader, max_batches=500, verbose=True):
        with tqdm(total=max_batches, disable=not verbose) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                pbar.set_postfix(loss='{0:.4f}'.format(results['mean_outer_loss']))

    def train_iter(self, dataloader, max_batches=500):
        num_batches = 0
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.model.train()
                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1

    def evaluate(self, dataloader, max_batches=500, verbose=True):
        with tqdm(total=max_batches, disable=not verbose) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                pbar.set_postfix(loss='{0:.4f}'.format(results['mean_outer_loss']))

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                self.model.eval()
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1
