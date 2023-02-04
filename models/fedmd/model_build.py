import torch.nn.functional as F

from torch import nn


def build_model(num_classes=100, n1=32, n2=64, n3=128, softmax=True):
    class ClientX(nn.Module):
        def __init__(
            self, device, *_, **__
        ):
            self.device = device
            super(ClientX, self).__init__()
            n0 = 3
            x_in = 32
            padding = 1
            kernel_size = 3
            stride = 1
            avg_pool_kernel_size = 2
            avg_pool_stride = 1
            self.CNN1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=n0,
                    kernel_size=kernel_size,
                    out_channels=n1,
                    padding=padding,
                    stride=stride,
                ),
                nn.BatchNorm2d(n1),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.ZeroPad2d(padding=(1, 0, 1, 0)),
                nn.AvgPool2d(kernel_size=avg_pool_kernel_size, stride=1),
            )
            conv2d_out = (x_in + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1
            zero_pad = conv2d_out + 1
            avgpool_out = (zero_pad - avg_pool_kernel_size) / avg_pool_stride + 1

            padding = 0
            kernel_size = 2
            stride = 2
            avg_pool_kernel_size = 2
            avg_pool_stride = 2
            x_in = avgpool_out
            self.CNN2 = nn.Sequential(
                nn.Conv2d(
                    in_channels=n1,
                    stride=stride,
                    kernel_size=kernel_size,
                    out_channels=n2,
                    padding=padding,
                ),
                nn.BatchNorm2d(n2),
                nn.ReLU(),
                nn.Dropout2d(0.2),
                nn.AvgPool2d(kernel_size=2),
            )
            conv2d_out = (x_in + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1
            avgpool_out = (conv2d_out - avg_pool_kernel_size) / avg_pool_stride + 1

            padding = 0
            kernel_size = 2
            stride = 2
            zero_pad = 0
            x_in = avgpool_out
            self.CNN3 = nn.Sequential(
                nn.Conv2d(
                    in_channels=n2,
                    stride=stride,
                    kernel_size=kernel_size,
                    out_channels=n3,
                ),
                nn.BatchNorm2d(n3),
                nn.ReLU(),
                nn.Dropout2d(0.2),
            )
            x_out = (x_in + 2 * padding - 1 * (kernel_size - 1) - 1) / stride + 1
            self.FC1 = nn.Linear(int(x_out * x_out * n3), num_classes)

        def forward(self, x):
            x = self.CNN1(x)
            # print(x.shape)
            x = self.CNN2(x)
            # print(x.shape)
            x = self.CNN3(x)
            # print(x.shape)
            x = x.view(x.shape[0], -1)  # 展开
            # print(x.shape)
            x = self.FC1(x)
            if softmax:
                return F.softmax(x, dim=1)
            else:
                return x

    return ClientX
