import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

from .ste import autograd_ge, autograd_lt
from .gradient import GradientOp


class ScalarLinear(nn.Module):
    def __init__(self, bias=0.0):
        super(ScalarLinear, self).__init__()
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, input):
        return input + self.bias


def align(a: torch.Tensor, b: torch.Tensor):
    '''
    make tensor b be the same size(in last 2 dimensions) with tensor a.
    :param a: A 4-dim tensor.
    :param b: A 4-dim tensor.
    :return:
    '''
    *_, ah, aw = a.shape
    *_, bh, bw = b.shape
    if ah != bh or aw != bw:
        b = b[:, :, :ah, :aw]

    return a, b


class CAConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, p_threshold=None):

        kernel_size = _pair(kernel_size)
        if kernel_size == (1, 1):
            raise ValueError("CAConv2d is not supported 1x1 size convolution kernel.")
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CAConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, False, _pair(0), groups, bias, 'zeros')

        self.p_threshold = p_threshold

        self.gradient_op = GradientOp(1, self.kernel_size, self.stride, self.padding, self.dilation)
        self.scalar_linear = ScalarLinear(bias=0.0)

        self._clear_statistics()
        self._update_complementary_weight()

    @property
    def proportion(self):
        return self.total_primary / (self.total_primary + self.total_secondary)

    def _update_complementary_weight(self):
        self.complementary_weight = self.weight.sum(dim=[2, 3], keepdim=True)

    def _clear_statistics(self):
        self.total_primary = 0
        self.total_secondary = 0

    def forward(self, feature):
        self._update_complementary_weight()

        with torch.no_grad():
            gradient = self.gradient_op(feature)  # (N,1,H_out*W_out)
        score = self.scalar_linear(gradient)

        if self.p_threshold is None:
            self.primary_mask = autograd_ge(score, 0)
            self.complementary_mask = autograd_lt(score, 0)
        else:
            probility = torch.sigmoid(score)
            self.primary_mask = (probility >= self.p_threshold).float()
            self.complementary_mask = (probility < self.p_threshold).float()

        self.total_primary += self.primary_mask.sum().item()
        self.total_secondary += self.complementary_mask.sum().item()

        primary_output = F.conv2d(feature, self.weight, self.bias, self.stride,
                                  self.padding, self.dilation, self.groups)
        complementary_output = F.conv2d(feature, self.complementary_weight, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)

        primary_output, complementary_output = align(primary_output, complementary_output)
        self.output = self.primary_mask * primary_output + self.complementary_mask * complementary_output
        return self.output

    def __repr__(self):
        return "CAConv2d(in_channels={}, out_channels={}, kernel_size={}, " \
               "stride={}, padding={}, dilation={}, groups={}, bias={})".format(
                   self.in_channels, self.out_channels, self.kernel_size,
                   self.stride, self.padding, self.dilation, self.groups, self.bias is not None
               )

    @staticmethod
    def from_conv2d(conv: nn.Conv2d, copy_weight=True):
        caconv2d = CAConv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
            conv.bias is not None,
        )
        if copy_weight:
            caconv2d.weight.data.copy_(conv.weight.data)
            if conv.bias is not None:
                caconv2d.bias.data.copy_(conv.bias.data)
        if hasattr(conv, "output_size"):
            caconv2d.output_size = conv.output_size
        return caconv2d
