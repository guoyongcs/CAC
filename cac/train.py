import os
from datetime import datetime

import pyhocon
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.engine import train, evaluate, LargerHolder
from core.metric import AverageMetric, AccuracyMetric
from core.model import cifar_resnet20
from core.loss import CACLoss
from core.utils import get_args, get_logger
from core.utils import set_cudnn_auto_tune
from core.utils import FLOPs
from core.utils import replace_convs_with_cac


if __name__ == "__main__":
    args = get_args()
    hocon = pyhocon.ConfigFactory.parse_file(args.config)
    output_directory = args.output_directory
    os.makedirs(output_directory, exist_ok=False)
    logger = get_logger("train", output_directory)

    set_cudnn_auto_tune()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=hocon.get_list("dataset.mean"),
            std=hocon.get_list("dataset.std"),
        ),
    ])
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=hocon.get_list("dataset.mean"),
            std=hocon.get_list("dataset.std"),
        ),
    ])
    trainset = torchvision.datasets.CIFAR10(root=hocon.get("dataset.root"), train=True,
                                            download=True, transform=train_transform)
    valset = torchvision.datasets.CIFAR10(root=hocon.get("dataset.root"), train=False,
                                          download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=hocon.get("strategy.train.batch_size"),
        shuffle=True, pin_memory=True, num_workers=hocon.get_int("machine.num_workers")
    )

    validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=hocon.get("strategy.validation.batch_size"),
        shuffle=False, pin_memory=True, num_workers=hocon.get_int("machine.num_workers")
    )

    net = cifar_resnet20(pretrained=hocon.get_bool("net.pretrained"),
                         num_classes=hocon.get_int("dataset.n_classes"))
    flops_tester = FLOPs(net, (1, 3, 32, 32))
    baseline_flops = flops_tester.compute()
    logger.info(f"The baseline FLOPs is {baseline_flops/1e6:.2f}M.")
    replace_convs_with_cac(net, logger=logger)
    net = net.to(device=device)
    from core.utils import network_bias

    criterion = CACLoss(
        original_loss=nn.CrossEntropyLoss(),
        lambda_=hocon.get_float("net.lambda"),
        baseline_flops=baseline_flops,
        flops_tester=flops_tester,
    )

    optimizer = optim.SGD(
        net.parameters(),
        lr=hocon.get_float("optimizer.learning_rate"),
        momentum=hocon.get_float("optimizer.momentum"),
        dampening=hocon.get_float("optimizer.dampening"),
        weight_decay=hocon.get_float("optimizer.weight_decay"),
        nesterov=hocon.get_bool("optimizer.nesterov")
    )

    max_epoch = hocon.get_int("strategy.epochs")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=hocon.get_list("scheduler.milestones"),
        gamma=hocon.get_float("scheduler.gamma")
    )
    writer = SummaryWriter(output_directory)

    log_frequency = hocon.get_int("machine.log_frequency")

    larger_holder = LargerHolder()

    estart = datetime.now()
    for epoch in range(1, max_epoch+1):
        train(epoch=epoch, data_loader=train_loader, device=device,
              model=net, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
              larger_holder=larger_holder, baseline_flops=baseline_flops, flops_tester=flops_tester,
              logger=logger, output_directory=output_directory,
              writer=writer, log_frequency=log_frequency)

        evaluate(epoch=epoch, data_loader=validation_loader, device=device,
                 model=net, criterion=criterion, larger_holder=larger_holder,
                 baseline_flops=baseline_flops, flops_tester=flops_tester,
                 logger=logger, output_directory=output_directory,
                 writer=writer,  log_frequency=log_frequency)
        logger.info(f"Epoch={epoch:03d} Complete, "
                    f"ETA={(datetime.now()-estart)/epoch*(max_epoch-epoch)}.")
    logger.info(f"Evaluate CIFAR10 Complete, eplased time={datetime.now()-estart}.")

    writer.close()
