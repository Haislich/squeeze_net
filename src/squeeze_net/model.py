"""Implementation of the SqueezeNet Model"""

import torch
from torch import nn


class Fire(nn.Module):
    """A Fire module is comprised of: a squeeze convolution layer
    (which has only 1x1 filters), feeding into an expand layer that has a mix of
    1x1 and 3x3 convolution filters.
    We expose three tunable dimensions (hyperparameters) in a Fire
    module: s1x1, e1x1, and e3x3.
    In a Fire module, s1x1 is the number of filters in the squeeze layer
    (all 1x1), e1x1 is the number of 1x1 filters in the expand layer, and e3x3 is the number of
    3x3 filters    in the expand layer.
    """

    def __init__(
        self,
        in_dim: int,
        squeeze1_dim: int,
        expand1_dim: int,
        expand3_dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert squeeze1_dim < (
            expand1_dim + expand3_dim
        ), "squeeze1_dim (s1x1), less than expand1_dim + expand3_dim (e1x1 + e3x3)"

        self.squeeze = nn.Conv2d(
            in_channels=in_dim,
            out_channels=squeeze1_dim,
            kernel_size=1,
        )
        self.squeeze_activation = nn.ReLU()

        self.expand1 = nn.Conv2d(
            in_channels=squeeze1_dim,
            out_channels=expand1_dim,
            kernel_size=1,
        )
        self.expand1_activation = nn.ReLU()

        self.expand3 = nn.Conv2d(
            in_channels=expand1_dim,
            out_channels=expand3_dim,
            kernel_size=3,
        )
        self.expand3_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the computation performed at every call"""
        x = self.squeeze_activation(self.squeeze(x))
        # The two activated outputs are concatenated along dimension 1,
        # which typically corresponds to the channel dimension.
        # Assuming that `x = [batch_size, num_channels, height, width]`
        return torch.cat(
            [
                self.expand1_activation(self.expand1(x)),
                self.expand3_activation(self.expand3(x)),
            ],
            dim=1,
        )


class SqueezeNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Images are assumed to be 224x224x3, WxHxC, and should return a tensor with 96 channels.
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=7,
            stride=2,
        )
        self.conv1_activation = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )

        self.fire2 = Fire(
            in_dim=96,
            squeeze1_dim=16,
            expand1_dim=64,
            expand3_dim=64,
        )
        self.fire3 = Fire(
            in_dim=128,
            squeeze1_dim=16,
            expand1_dim=64,
            expand3_dim=64,
        )
        self.fire14 = Fire(
            in_dim=128,
            squeeze1_dim=32,
            expand1_dim=128,
            expand3_dim=128,
        )

        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fire5 = Fire(
            in_dim=256,
            squeeze1_dim=32,
            expand1_dim=128,
            expand3_dim=128,
        )
        self.fire6 = Fire(
            in_dim=256,
            squeeze1_dim=48,
            expand1_dim=192,
            expand3_dim=192,
        )
        self.fire7 = Fire(
            in_dim=384,
            squeeze1_dim=48,
            expand1_dim=192,
            expand3_dim=192,
        )
