import argparse
import functools
import logging
import os
import random
import sys

import numpy
import torch
import torch.nn as nn

from core.caconv import CAConv2d


def set_cudnn_auto_tune():
    torch.backends.cudnn.benchmark = True


def replace_layer_by_unique_name(module, unique_name, layer):
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        module._modules[unique_names[0]] = layer
    else:
        replace_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
            layer
        )


def get_layer_by_unique_name(module, unique_name) -> nn.Module:
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        return module._modules[unique_names[0]]
    else:
        return get_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
        )


def replace_convs_with_cac(net: nn.Module, copy_weight=True, logger=None) -> int:
    n_caconv = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            if module.kernel_size != (1, 1):
                if logger is not None:
                    logger.info(f"Replace Conv2d({name}) with CAConv2d, {module}.")
                caconv = CAConv2d.from_conv2d(module, copy_weight=copy_weight)
                replace_layer_by_unique_name(net, name, caconv)
                n_caconv += 1
    if logger is not None:
        logger.info(f"Replace complete.({n_caconv} Conv2d layers have been replaced.)")
    return n_caconv


def network_proportion(net: nn.Module) -> float:
    proportions = []
    for m in net.modules():
        if isinstance(m, CAConv2d):
            proportions.append(f"{m.proportion:.2f}")
    return proportions


def network_bias(net: nn.Module) -> float:
    bias = []
    for m in net.modules():
        if isinstance(m, CAConv2d):
            bias.append(m.scalar_linear.bias.item())
    return bias


def clear_statistics(net: nn.Module):
    for m in net.modules():
        if isinstance(m, CAConv2d):
            m._clear_statistics()


def get_logger(name: str, output_directory: str,
               file_name: str = "default.log", debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, file_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger


def get_args(argv=sys.argv) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train a network")
    parser.add_argument("-c", "--config", type=str, help="the path to config file.", default=None)
    parser.add_argument("-o", "--output_directory", type=str, help="the path to store experiment files.", default=None)
    args, _ = parser.parse_known_args(argv)
    return args


def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    *_, h, w = output.shape
    module.output_size = (h, w)


class FLOPs(object):
    def __init__(self, net: nn.Module, size):
        self.net = net
        self.size = size
        hooks = []
        for name, m in self.net.named_modules():
            if isinstance(m, (nn.Conv2d, CAConv2d)):
                hooks.append(m.register_forward_hook(size_hook))
        with torch.no_grad():
            training = self.net.training
            self.net.eval()
            self.net(torch.rand(self.size))
            self.net.train(mode=training)
        for hook in hooks:
            hook.remove()

    def compute(self, last_batch=False):
        flops = 0
        for name, m in self.net.named_modules():
            if isinstance(m, CAConv2d):
                h, w = m.output_size
                kh, kw = m.kernel_size
                tmp = h * w * m.in_channels * m.out_channels * kh * kw / m.groups
                if last_batch:
                    mask = m.primary_mask
                    p = mask.sum() / mask.numel()
                else:
                    p = m.proportion
                alpha = p + (1 - p) / (kh * kw)
                flops += alpha * tmp + 13 * h * w
            if isinstance(m, nn.Conv2d):
                h, w = m.output_size
                kh, kw = m.kernel_size
                flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
            if isinstance(m, nn.Linear):
                flops += m.in_features * m.out_features
        return flops
