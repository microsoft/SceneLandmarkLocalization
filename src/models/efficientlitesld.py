import torch
import torch.nn as nn

from .blocks import _make_encoder


class ASPP(nn.Module):
    def __init__(self, in_ch, d1, d2, d3, d4, reduction=4):
        super(ASPP, self).__init__()
        self.aspp_d1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d1, dilation=d1),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )
        self.aspp_d2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d2, dilation=d2),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )
        self.aspp_d3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d3, dilation=d3),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )

        self.aspp_d4 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // reduction, 3, padding=d4, dilation=d4),
            nn.BatchNorm2d(in_ch // reduction),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.aspp_d1(x)
        d2 = self.aspp_d2(x)
        d3 = self.aspp_d3(x)
        d4 = self.aspp_d4(x)
        return torch.cat((d1, d2, d3, d4), dim=1)


class EfficientNetSLD(torch.nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path=None, num_landmarks=200, output_downsample=4, features=320):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
            backbone (str, optional): Backbone network for encoder. Defaults to efficientnetlite0
        """
        super(EfficientNetSLD, self).__init__()

        self.pretrained, _ = _make_encoder(use_pretrained=True, output_downsample=output_downsample)

        self.aspp = nn.Sequential(
            ASPP(in_ch=features, d1=1, d2=2, d3=3, d4=4, reduction=4),
        )

        self.heatmap_outputs_res1 = nn.Sequential(
            nn.Conv2d(features, num_landmarks, kernel_size=1, stride=1, padding=0)
        )
        self.heatmap_outputs_res2 = None

        if output_downsample == 2:
            input_channels = features + num_landmarks
            output_channels = features

            self.heatmap_features_res2 = nn.Sequential(nn.ConvTranspose2d(in_channels=input_channels,
                                                                          out_channels=output_channels,
                                                                          kernel_size=4, stride=2, padding=1,
                                                                          bias=False),
                                                       nn.BatchNorm2d(output_channels),
                                                       nn.ReLU(inplace=True)
                                                       )
            self.heatmap_outputs_res2 = nn.Conv2d(output_channels, num_landmarks, kernel_size=1, stride=1, bias=False)

        if path:
            self.load(path)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            Heatmap prediction
            ['1']: quarter of input spatial dimension
            ['2']: half of input spatial dimension
        """

        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)
        y1 = self.aspp(layer_4)
        z1 = self.heatmap_outputs_res1(y1)

        z2 = None
        if self.heatmap_outputs_res2 is not None:
            y2 = self.heatmap_features_res2(torch.cat((y1, z1), dim=1))
            z2 = self.heatmap_outputs_res2(y2)

        return {'1': z1, '2': z2}
