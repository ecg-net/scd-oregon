from torch import nn
import torch
from functools import reduce
from operator import __add__
import torch.nn.functional as F
from collections import OrderedDict
from typing import Callable, List
from torch import Tensor


class EffNet(nn.Module):
    def __init__(
        self,
        num_additional_features: int = 0,
        output_neurons: int = 1,
        channels: List[int] = (32, 16, 24, 40, 80, 112, 192, 320, 1280),
        depth: List[int] = (1, 2, 2, 3, 3, 3, 3),
        dilation: int = 2,
        stride: int = 8,
        expansion: int = 6,
        embedding_hook: bool = False,
        input_channels: int = 12,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.channels = channels
        self.output_nerons = output_neurons

        self.depth = depth
        self.expansion = expansion
        self.stride = stride
        self.dilation = dilation
        self.embedding_hook = embedding_hook

        self.stage1 = nn.Conv1d(
            self.input_channels,
            self.channels[0],
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
        )  # 1 conv

        self.b0 = nn.BatchNorm1d(self.channels[0])

        self.stage2 = MBConv(
            self.channels[0], self.channels[1], self.expansion, self.depth[0], stride=2
        )

        self.stage3 = MBConv(
            self.channels[1], self.channels[2], self.expansion, self.depth[1], stride=2
        )

        self.Pool = nn.MaxPool1d(3, stride=1, padding=1)

        self.stage4 = MBConv(
            self.channels[2], self.channels[3], self.expansion, self.depth[2], stride=2
        )

        self.stage5 = MBConv(
            self.channels[3], self.channels[4], self.expansion, self.depth[3], stride=2
        )

        self.stage6 = MBConv(
            self.channels[4], self.channels[5], self.expansion, self.depth[4], stride=2
        )

        self.stage7 = MBConv(
            self.channels[5], self.channels[6], self.expansion, self.depth[5], stride=2
        )

        self.stage8 = MBConv(
            self.channels[6], self.channels[7], self.expansion, self.depth[6], stride=2
        )

        self.stage9 = nn.Conv1d(self.channels[7], self.channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(
            self.channels[8] + num_additional_features, self.output_nerons
        )
        self.fc.bias.data[0] = 0.275

    def forward(self, x: Tensor) -> Tensor:
        if self.num_additional_features > 0:
            x, additional = x

        x = self.b0(self.stage1(x))
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.Pool(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.Pool(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = self.act(self.AAP(x)[:, :, 0])
        x = self.drop(x)

        if self.num_additional_features > 0:
            x = torch.cat((x, additional), 1)

        if self.embedding_hook:
            return x
        else:
            x = self.fc(x)
            return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        conv_padding = reduce(
            __add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]],
        )
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


def output_channels(num):
    return int(num / 3) + int(num) + int(num) + int(num / 3)


class Multi_2D_CNN_block(nn.Module):
    def __init__(self, in_channels, num_kernel):
        super().__init__()

        conv_block = BasicConv2d
        self.a = conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1))

        self.b = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 2), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3)),
        )

        self.c = nn.Sequential(
            conv_block(in_channels, int(num_kernel / 3), kernel_size=(1, 1)),
            conv_block(int(num_kernel / 3), int(num_kernel / 2), kernel_size=(3, 3)),
            conv_block(int(num_kernel / 2), int(num_kernel), kernel_size=(3, 3)),
        )

        self.d = nn.Sequential(
            nn.Conv2d(in_channels, int(num_kernel / 3), kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), padding=(1, 1), stride=1),
        )

        self.out_channels = output_channels(num_kernel)

        self.bn = nn.BatchNorm2d(self.out_channels)

    def get_out_channels(self):
        return self.out_channels

    def forward(self, x):
        branch1 = self.a(x)
        branch2 = self.b(x)
        branch3 = self.c(x)
        branch4 = self.d(x)
        output = [branch1, branch2, branch3, branch4]

        # BatchNorm across the concatenation of output channels from final layer of Branch 1/2/3
        # ,1 refers to the channel dimension
        final = self.bn(torch.cat(output, 1))
        return final


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()

        conv_padding = reduce(
            __add__,
            [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in kernel_size[::-1]],
        )
        self.pad = nn.ZeroPad2d(conv_padding)
        # ZeroPad2d(2) would add 2 to all 4 sides, ZeroPad2d((1,1,2,0)) would add 1 left, 1 right, 2 above, 0 below

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=False, **kwargs
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        expansion: int,
        activation: Callable,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()

        self.stride = stride
        self.conv1 = nn.Conv1d(in_channel, in_channel * expansion, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channel * expansion,
            in_channel * expansion,
            kernel_size=3,
            groups=in_channel * expansion,
            padding=padding,
            stride=stride,
        )
        self.conv3 = nn.Conv1d(
            in_channel * expansion, out_channel, kernel_size=1, stride=1
        )
        self.b0 = nn.BatchNorm1d(in_channel * expansion)
        self.b1 = nn.BatchNorm1d(in_channel * expansion)
        self.d = nn.Dropout()
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x + y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y


class MBConv(nn.Module):
    def __init__(
        self, in_channel, out_channels, expansion, layers, activation=nn.ReLU6, stride=2
    ):
        super().__init__()

        self.stack = OrderedDict()
        for i in range(0, layers - 1):
            self.stack["s" + str(i)] = Bottleneck(
                in_channel, in_channel, expansion, activation
            )

        self.stack["s" + str(layers + 1)] = Bottleneck(
            in_channel, out_channels, expansion, activation, stride=stride
        )

        self.stack = nn.Sequential(self.stack)

        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stack(x)
        return self.bn(x)
