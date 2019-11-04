import os
from datetime import datetime
from functools import partial
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.metric import AverageMetric, AccuracyMetric
from core.utils import FLOPs, network_proportion, network_bias
from core.utils import clear_statistics


class LargerHolder:
    def __init__(self):
        self.value = 0.0
        self.metadata = dict(epoch=-1)

    def update(self, new_value, metadata):
        if new_value > self.value:
            self.value = new_value
            self.metadata = metadata
            return True
        else:
            return False


def _iter_impl(epoch: int, phase: str, data_loader: DataLoader, device: str,
               model: nn.Module, criterion: nn.Module,
               optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
               larger_holder: LargerHolder, baseline_flops: float, flops_tester: FLOPs,
               logger: logging.Logger, output_directory: str,
               writer: SummaryWriter, log_frequency: int):
    start = datetime.now()

    clear_statistics(model)
    model.train(phase == "train")
    loss_metric = AverageMetric()
    accuracy_metric = AccuracyMetric(topk=(1, 5))

    for iter_, (datas, targets) in enumerate(data_loader, start=1):
        datas, targets = datas.to(device=device), targets.to(device=device)
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(datas)
        loss = criterion(outputs, targets)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_metric.update(loss)
        accuracy_metric.update(targets, outputs)

        if iter_ % log_frequency == 0:
            logger.info(f"{phase.upper()}, epoch={epoch:03d}, iter={iter_}/{len(data_loader)}, "
                        f"loss={loss_metric.last:.4f}({loss_metric.value:.4f}), "
                        f"accuracy@1={accuracy_metric.last_accuracy(1).rate*100:.2f}%"
                        f"({accuracy_metric.accuracy(1).rate*100:.2f}%), "
                        f"accuracy@5={accuracy_metric.last_accuracy(5).rate*100:.2f}%"
                        f"({accuracy_metric.accuracy(5).rate*100:.2f}%), ")

    if phase != "train":
        acc = accuracy_metric.accuracy(1).rate
        if larger_holder.update(new_value=acc, metadata=dict(epoch=epoch)):
            if output_directory is not None:
                torch.save(model.state_dict(), os.path.join(output_directory, "best_model.pth"))

    if scheduler is not None:
        scheduler.step()

    flops = flops_tester.compute()
    logger.info(f"{phase.upper()} Complete, epoch={epoch:03d}, "
                f"loss={loss_metric.value:.4f}, "
                f"accuracy@1={accuracy_metric.accuracy(1).rate*100:.2f}%, "
                f"accuracy@5={accuracy_metric.accuracy(5).rate*100:.2f}%, "
                f"flops={flops/1e6:.2f}M({flops/baseline_flops*100:.2f}%), "
                f"best_accuracy={larger_holder.value*100:.2f}%(epoch={larger_holder.metadata['epoch']:03d}), "
                f"propotions={network_proportion(model)}, "
                f"eplased time={datetime.now()-start}.")

    writer.add_scalar(f"{phase}/loss", loss_metric.value, epoch)
    writer.add_scalar(f"{phase}/accuracy@1", accuracy_metric.accuracy(1).rate, epoch)
    writer.add_scalar(f"{phase}/accuracy@5", accuracy_metric.accuracy(5).rate, epoch)


train = partial(_iter_impl, phase="train")
evaluate = partial(_iter_impl, phase="validation", optimizer=None, scheduler=None)
