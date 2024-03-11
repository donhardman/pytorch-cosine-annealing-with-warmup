import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.,
                 max_lr: list = None,  # Changed to list
                 min_lr: list = None,  # Changed to list
                 warmup_steps: int = 0,
                 gamma: float = 1.,
                 last_epoch: int = -1):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr if max_lr is not None else [group['lr'] for group in optimizer.param_groups]
        self.max_lr = list(self.base_max_lr)  # Copy to ensure it's a separate list
        self.min_lr = min_lr if min_lr is not None else [0.001] * len(optimizer.param_groups)  # Default min_lr if not provided
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # Initialize learning rate
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group, min_lr in zip(self.optimizer.param_groups, self.min_lr):
            param_group['lr'] = min_lr
            self.base_lrs.append(min_lr)

    def get_last_lr(self):
        return self.get_lr()

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]
        else:
            return [base_lr + (max_lr - base_lr) * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) / (self.cur_cycle_steps - self.warmup_steps))) / 2 for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.step_in_cycle = epoch if epoch < self.first_cycle_steps else (epoch - self.first_cycle_steps) % self.cur_cycle_steps + 1
        if self.step_in_cycle == 0: self.step_in_cycle = self.cur_cycle_steps
        if epoch >= self.first_cycle_steps:
            cycle = (epoch - self.first_cycle_steps) // self.cur_cycle_steps + 1
            self.cur_cycle_steps = self.first_cycle_steps * (self.cycle_mult ** cycle)
            self.max_lr = [base_max_lr * (self.gamma ** cycle) for base_max_lr in self.base_max_lr]
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
