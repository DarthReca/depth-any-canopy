# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


import math

import torch
import torchvision
from torch import nn

from .backbone import resize


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.
    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.
    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.
    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        with_spectral_norm (bool): Whether use spectral norm in conv module.
            Default: False.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = "conv_block"

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias="auto",
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="ReLU"),
        inplace=True,
        with_spectral_norm=False,
        padding_mode="zeros",
        order=("conv", "norm", "act"),
    ):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ["zeros", "circular"]
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {"conv", "norm", "act"}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == "auto":
            bias = not self.with_norm
        self.with_bias = bias

        if self.with_explicit_padding:
            pad_cfg = dict(type=padding_mode)
            self.padding_layer = build_padding_layer(pad_cfg, padding)
        # to do Camille put back

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = nn.Conv2d(  # build_conv_layer(#conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.norm_name = None  # type: ignore

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_["type"] not in [
                "Tanh",
                "PReLU",
                "Sigmoid",
                "HSigmoid",
                "Swish",
                "GELU",
            ]:
                act_cfg_.setdefault("inplace", inplace)
            self.activate = nn.ReLU()  # build_activation_layer(act_cfg_)

        # Use msra init by default
        torch.manual_seed(1)
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, "init_weights"):
            if self.with_activation and self.act_cfg["type"] == "LeakyReLU":
                nonlinearity = "leaky_relu"
                a = self.act_cfg.get("negative_slope", 0.01)
            else:
                nonlinearity = "relu"
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(
        self,
        x: torch.Tensor,
        activate: bool = True,
        norm: bool = True,
        debug: bool = False,
    ) -> torch.Tensor:
        for layer in self.order:
            if debug == True:
                breakpoint()
            if layer == "conv":
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                x = self.conv(x)
            elif layer == "norm" and norm and self.with_norm:
                x = self.norm(x)
            elif layer == "act" and activate and self.with_activation:
                x = self.activate(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


class HeadDepth(nn.Module):
    def __init__(self, features, classify=False, n_bins=256):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(
                32, 1 if not classify else n_bins, kernel_size=1, stride=1, padding=0
            ),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_channels=1024,  # 768,
        out_channels=[128, 256, 512, 1024],  # [96, 192, 384, 768],
        readout_type="project",  # 'ignore',
        patch_size=16,
    ):
        super(ReassembleBlocks, self).__init__()  # init_cfg)

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg=None,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )
                # build_activation_layer(dict(type='GELU'))))

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_cfg (dict): dictionary to construct and config activation layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self, in_channels, act_cfg, norm_cfg, stride=1, dilation=1, init_cfg=None
    ):
        super(PreActResidualConvUnit, self).__init__()  # init_cfg)
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )
        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self,
        in_channels,
        act_cfg,
        norm_cfg,
        expand=False,
        align_corners=True,
        init_cfg=None,
    ):
        super(FeatureFusionBlock, self).__init__()  # init_cfg)
        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners
        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2
        self.project = ConvModule(
            self.in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True
        )
        self.res_conv_unit1 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg
        )
        self.res_conv_unit2 = PreActResidualConvUnit(
            in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg
        )

    def forward(self, *inputs):
        x = inputs[0]

        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(
                    inputs[1],
                    size=(x.shape[2], x.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


class DPTHead(nn.Module):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """

    def __init__(
        self,
        in_channels=(1024, 1024, 1024, 1024),
        channels=256,
        embed_dims=1024,
        post_process_channels=[128, 256, 512, 1024],
        readout_type="project",
        patch_size=16,
        expand_channels=False,
        min_depth=0.001,
        classify=False,
        n_bins=256,
        **kwargs,
    ):
        super(DPTHead, self).__init__(**kwargs)
        torch.manual_seed(1)
        self.channels = channels
        self.norm_cfg = None
        self.min_depth = min_depth
        self.max_depth = 10
        self.n_bins = n_bins
        self.classify = classify
        self.in_channels = in_channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(
            in_channels=embed_dims,  # Camille 23-06-26
            out_channels=post_process_channels,
        )  # Camille 23-06-26

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel
            for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(
                ConvModule(
                    channel,
                    self.channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=None,
                    bias=False,
                )
            )
        self.fusion_blocks = nn.ModuleList()
        self.act_cfg = {"type": "ReLU"}
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(
                FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg)
            )
        self.fusion_blocks[0].res_conv_unit1 = None
        torch.manual_seed(1)
        self.project = ConvModule(
            self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=None
        )
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels
        # self.conv_depth = HeadDepth(self.channels)
        self.conv_depth = HeadDepth(self.channels, self.classify, self.n_bins)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]

        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])

        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])

        out = self.project(out)
        if self.classify:
            logit = self.conv_depth(out)

            # if self.bins_strategy == 'UD':
            bins = torch.linspace(
                self.min_depth, self.max_depth, self.n_bins, device=inputs[0][0].device
            )
            # linear strategy
            logit = torch.relu(logit)
            eps = 0.1
            logit = logit + eps
            logit = logit / logit.sum(dim=1, keepdim=True)
            out = torch.einsum("ikmn,k->imn", [logit, bins]).unsqueeze(
                dim=1
            )  # + self.min_depth
        else:
            out = self.relu(self.conv_depth(out)) + self.min_depth

        return out
