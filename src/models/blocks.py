import torch
import torch.nn as nn
from .conv2d_layers import Conv2dSameExport


def _make_encoder(use_pretrained, exportable=True, output_downsample=4):

    # pretrained = _make_pretrained_efficientnet_lite0(use_pretrained, exportable=exportable)
    pretrained = torch.load('pretrained_efficientnetlite0.net')

    if output_downsample <= 16:
        pretrained.layer2[0][0].conv_dw.stride = (1, 1)
    if output_downsample <= 8:
        pretrained.layer3[0][0].conv_dw.stride = (1, 1)
    if output_downsample <= 4:
        pretrained.layer4[0][0].conv_dw.stride = (1, 1)

    return pretrained, None


def _make_pretrained_efficientnet_lite0(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite0",
        pretrained=use_pretrained,
        exportable=exportable
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained


def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained


def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
