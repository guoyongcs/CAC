import torch
import torch.nn as nn

from core.caconv import CAConv2d


class CACLoss:
    def __init__(self, original_loss, lambda_, baseline_flops, flops_tester):
        self.original_loss = original_loss
        self.lambda_ = lambda_
        self.baseline_flops = baseline_flops
        self.flops_tester = flops_tester

    def __call__(self, *args, **kwargs):
        loss = self.original_loss(*args, **kwargs)
        loss = loss * torch.pow(self.flops_tester.compute(last_batch=True)/self.baseline_flops, self.lambda_)
        return loss
