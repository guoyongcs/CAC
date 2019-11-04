import torch
import torch.nn as nn
import torch.nn.functional as F

GradientKernels = {
    (1, 2): torch.tensor([[1.0, -1.0]], dtype=torch.float32),
    (2, 1): torch.tensor([[1.0], [-1.0]], dtype=torch.float32),

    (1, 3): torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32),
    (3, 1): torch.tensor([[1.0], [0.0], [-1.0]], dtype=torch.float32),

    (2, 2): (torch.tensor([[1.0, -1.0],
                           [1.0, -1.0]], dtype=torch.float32),
             torch.tensor([[1.0, 1.0],
                           [-1.0, -1.0]], dtype=torch.float32),
             ),
    (2, 3): (torch.tensor([[1.0, 0.0, -1.0],
                           [1.0, 0.0, -1.0]], dtype=torch.float32),
             torch.tensor([[1.0, 1.0, 1.0],
                           [-1.0, -1.0, -1.0]], dtype=torch.float32),
             ),
    (3, 2): (torch.tensor([[1., -1.],
                           [1., -1.],
                           [1., -1.]], dtype=torch.float32),
             torch.tensor([[1., 1.],
                           [0., 0.],
                           [-1., -1.]], dtype=torch.float32),
             ),

    (3, 3): (torch.tensor([[1., 0., -1.],
                           [2., 0., -2.],
                           [1., 0., -1.]], dtype=torch.float32),
             torch.tensor([[1., 2., 1.],
                           [0., 0., 0.],
                           [-1., -2., -1.]], dtype=torch.float32)
             ),
}


def get_prewitt_kernel(size):
    return GradientKernels[size]


def is_odd(number):
    return number % 2 == 1


def one_size_down_kernel_size(size):
    if size <= 3:
        return size
    else:
        if is_odd(size):
            return 3
        else:
            return 2


def down_kernel_size(kernel_size):
    hkernel, wkernel = kernel_size
    return one_size_down_kernel_size(hkernel), one_size_down_kernel_size(wkernel)


def one_side_pool_size(original_size, down_size, stride, dilation):
    return dilation * (original_size - down_size) // stride


def get_pool_size(original_size, down_size, stride, dilation):
    original_hsize, original_wsize = original_size
    down_hsize, down_wsize = down_size
    hstride, wstride = stride
    hdilation, wdilation = dilation
    pool_hsize, pool_wsize = one_side_pool_size(original_hsize, down_hsize, hstride, hdilation), \
        one_side_pool_size(original_wsize, down_wsize, wstride, wdilation)
    if pool_hsize == 0 and pool_wsize == 0:
        return None
    else:
        pool_hsize += 1
        pool_wsize += 1
        return (pool_hsize, pool_wsize)


class GradientOp(nn.Module):
    '''
    It is for computing the image gradient (or feature maps gradient).
    '''

    def __init__(self, in_channels, kernel_size, stride, padding, dilation):
        super(GradientOp, self).__init__()
        self.in_channels = in_channels
        if self.in_channels != 1:
            raise ValueError(
                "The in_channels should be 1, but got {}.".format(self.in_channels))
        self.kernel_size = kernel_size
        self.down_kernel_size = down_kernel_size(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.is_1d_kernel = 1 in self.down_kernel_size

        if self.is_1d_kernel:
            weight = get_prewitt_kernel(self.down_kernel_size)
            weight = weight.view(1, 1, *self.down_kernel_size)
            self.register_buffer("weight", weight)
        else:
            weight_x, weight_y = get_prewitt_kernel(self.down_kernel_size)
            weight_x, weight_y = weight_x.view(1, 1, *self.down_kernel_size), \
                weight_y.view(1, 1, *self.down_kernel_size)
            self.register_buffer("weight_x", weight_x)
            self.register_buffer("weight_y", weight_y)

    def forward(self, feature):
        with torch.no_grad():
            feature = feature.mean(dim=1, keepdim=True)  # (N,1,H_out,W_out)
            if self.is_1d_kernel:
                gradient = F.conv2d(feature, self.weight, None, self.stride,
                                    self.padding, self.dilation, self.in_channels)
                gradient.abs_()
            else:
                gradient_x = F.conv2d(feature, self.weight_x, None, self.stride,
                                      self.padding, self.dilation, self.in_channels)
                gradient_y = F.conv2d(feature, self.weight_y, None, self.stride,
                                      self.padding, self.dilation, self.in_channels)
                gradient = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

            pool_size = get_pool_size(
                self.kernel_size, self.down_kernel_size, self.stride, self.dilation)
            if pool_size is not None:
                gradient = F.avg_pool2d(
                    gradient, pool_size, stride=1, padding=0)
            N, C, H, W = gradient.shape
            if C != 1:
                raise Exception(
                    "The number of gradient channels should be 1, but got {}.".format(C))
            gradient = gradient.view(N, 1, H, W)
            return gradient

    def extra_repr(self):
        s = "in_channels={in_channels}, kernel_size={kernel_size}, stride={stride}, " \
            "padding={padding}, dilation={dilation}"
        return s.format(**self.__dict__)
