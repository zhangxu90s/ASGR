# Copyright 2026 Zijiang Yang.
# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

import copy
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# =========================
# Backbones
# =========================
# ---- ShuffleNetV2 ----
model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",
    "shufflenetv2_1.5x": "https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth",
    "shufflenetv2_2.0x": "https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth",
}


def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    channels_per_group = c // groups
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, activation="ReLU"):
        super(ShuffleV2Block, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),
                nn.Conv2d(
                    inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(branch_features),
                act_layers(activation),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(
                branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            act_layers(activation),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        model_size="1.5x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        kernal_size=3,
        activation="ReLU",
        pretrain=True,
        *args,
        **kwargs
    ):
        super(ShuffleNetV2, self).__init__()
        assert set(out_stages).issubset((2, 3, 4))
        print("ShuffleNetV2 model size:", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        self.activation = activation
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            act_layers(activation),
        )
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [ShuffleV2Block(input_channels, output_channels, 2, activation)]
            for _ in range(repeats - 1):
                seq.append(ShuffleV2Block(output_channels, output_channels, 1, activation))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                act_layers(activation),
            )
            self.stage4.add_module("conv5", conv5)
        self._initialize_weights(pretrain)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        outputs = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                outputs.append(x)
        return tuple(outputs)

    def _initialize_weights(self, pretrain=True):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if pretrain:
            url = model_urls.get(f"shufflenetv2_{self.model_size}")
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)
                print("=> loading pretrained model", url)
                self.load_state_dict(pretrained_state_dict, strict=False)


# ---- ResNet50 ----
try:
    from torchvision.models import ResNet50_Weights, resnet50
    _HAS_TV_R50_WEIGHTS = True
except Exception:
    from torchvision.models import resnet50  # type: ignore
    ResNet50_Weights = None  # type: ignore
    _HAS_TV_R50_WEIGHTS = False


class ResNet50(nn.Module):
    def __init__(self, out_stages=(2, 3, 4), pretrain=True, *args, **kwargs):
        super(ResNet50, self).__init__()
        assert set(out_stages).issubset((1, 2, 3, 4))

        if _HAS_TV_R50_WEIGHTS and pretrain:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            try:
                backbone = resnet50(pretrained=pretrain)  # old API
            except TypeError:
                backbone = resnet50(weights=None)

        self.out_stages = out_stages
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outputs = []
        x = self.layer1(x)
        if 1 in self.out_stages:
            outputs.append(x)
        x = self.layer2(x)
        if 2 in self.out_stages:
            outputs.append(x)
        x = self.layer3(x)
        if 3 in self.out_stages:
            outputs.append(x)
        x = self.layer4(x)
        if 4 in self.out_stages:
            outputs.append(x)
        return tuple(outputs)




class RobustMorphologyFusion(nn.Module):

    def __init__(self, in_channels_list, out_channels):
        super(RobustMorphologyFusion, self).__init__()
        self.num_scales = len(in_channels_list)

        # 1. 各尺度特征独立映射与量级对齐
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, 1, bias=False),
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
            for c in in_channels_list
        ])

        # 2. 跨尺度上下文提取器 (利用 Element-wise 相加降低混淆)
        self.context_extractor = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # 3. 空间自适应融合权重预测器 (输出 [B, num_scales, H, W])
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, self.num_scales, kernel_size=1, bias=True)
        )

        # 4. 通道自适应校准器 (输出 [B, num_scales, C, 1, 1])
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // 2, self.num_scales * out_channels, 1, bias=True
            ),
        )

        # 5. 后处理与残差门控
        self.post_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
        )

        # 可学习残差缩放参数，初始值给 1.0，保证训练初期不破坏基线
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, features):
        target_size = features[0].shape[2:]
        proj_feats = []

        # 对齐空间尺寸与通道数
        for i, feat in enumerate(features):
            p = self.projs[i](feat)
            if p.shape[2:] != target_size:
                p = F.interpolate(
                    p, size=target_size, mode="bilinear", align_corners=False
                )
            proj_feats.append(p)

        # 堆叠特征 [B, N, C, H, W]
        stacked_feats = torch.stack(proj_feats, dim=1)

        # 1. 计算融合上下文基底 (无通道混淆的相加基底) [B, C, H, W]
        base_context = torch.sum(stacked_feats, dim=1)
        context_feat = self.context_extractor(base_context)

        # 2. 空间维度：计算各尺度的 Softmax 竞合权重 [B, N, 1, H, W]
        spatial_weights = F.softmax(
            self.spatial_gate(context_feat), dim=1
        ).unsqueeze(2)

        # 3. 通道维度：计算 Sigmoid 通道感知校准因子 [B, N, C, 1, 1]
        B, C, H, W = context_feat.shape
        channel_weights = torch.sigmoid(
            self.channel_gate(context_feat).view(
                B, self.num_scales, C, 1, 1
            )
        )

        # 4. 特征融合：(特征 * 通道缩放) 进行 空间 Softmax 加权求和
        # 这种设计保证了特征量级始终在合理区间内
        calibrated_feats = stacked_feats * channel_weights
        fused_feat = torch.sum(
            calibrated_feats * spatial_weights, dim=1
        )  # [B, C, H, W]

        # 5. 残差连接：原始相加特征 + 可学习系数 * 融合特征
        output = base_context + self.gamma * self.post_fuse(fused_feat)

        return self.act(output)

# =====================================================================
# 1. 病理各向异性切比雪夫图卷积 (专治细胞核挤压/纺锤形变形)
# =====================================================================
class NucleiAnisotropicChebyshevConv(nn.Module):
    """
    针对细胞核重叠与异型性设计的多方向切比雪夫谱域卷积。
    包含 0°, 45°, 90°, 135° 四个方向的非对称滤波算子。
    """

    def __init__(self, channels: int):
        super().__init__()
        # 1. 四个方向的 1D / 2D 专用邻域卷积，捕捉异型细胞核的轴向边缘
        self.conv_horizontal = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False
        )
        self.conv_vertical = nn.Conv2d(
            channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False
        )
        self.conv_diag = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        # 方向融合与切比雪夫变换器 (T0, T1, T2)
        self.cheb_0 = nn.Conv2d(channels, channels, 1, bias=False)
        self.cheb_1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.cheb_2 = nn.Conv2d(channels, channels, 1, bias=False)

        self.gn = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T0: 0阶切比雪夫基底
        t0 = x

        # T1: 融合多方向各向异性邻域信息 (A * X)
        dir_h = self.conv_horizontal(x)
        dir_v = self.conv_vertical(x)
        dir_d = self.conv_diag(x)
        t1 = (dir_h + dir_v + dir_d) / 3.0

        # T2: 2阶高阶谱域递归展开 (2 * A * T1 - T0)
        t2 = 2.0 * ((self.conv_horizontal(t1) + self.conv_vertical(t1) + self.conv_diag(t1)) / 3.0) - t0

        # 谱域频域聚合
        out = self.cheb_0(t0) + self.cheb_1(t1) + self.cheb_2(t2)
        out = self.gn(out)
        return self.act(x + out)


# =====================================================================
# 3. 针对密集细胞核检测专用的 PointCloudGNNDetector
# =====================================================================
class PathologyNucleiGNNDetector(nn.Module):

    def __init__(self, in_channels_list, feat_channels=96, num_classes=2, space=8):
        super().__init__()
        self.space = space

        # 1. 跨尺度形态学融合 (使用原有 RobustMorphologyFusion)
        self.fusion = RobustMorphologyFusion(in_channels_list, feat_channels)

        # 2. 病理各向异性切比雪夫图神经网络
        self.gnn1 = NucleiAnisotropicChebyshevConv(feat_channels)
        self.gnn2 = NucleiAnisotropicChebyshevConv(feat_channels)

        # 4. 新增：细胞核中心概率预测头 (Center Heatmap Head)，辅助解耦粘连核
        self.center_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=feat_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # 5. 分类与回归头
        self.cls_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=feat_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=feat_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, num_classes, 1),
        )

        self.reg_head = nn.Sequential(
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=feat_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=feat_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feat_channels, 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.reg_head[-1].weight, std=0.001)
        if self.reg_head[-1].bias is not None:
            nn.init.constant_(self.reg_head[-1].bias, 0)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.normal_(self.cls_head[-1].weight, std=0.01)
        if self.cls_head[-1].bias is not None:
            nn.init.constant_(self.cls_head[-1].bias, bias_value)

    def _generate_anchors(self, images, feat_h, feat_w):
        bs, _, h, w = images.shape
        dev, dt = images.device, images.dtype

        grid_y, grid_x = torch.meshgrid(
            torch.arange(feat_h, device=dev, dtype=dt),
            torch.arange(feat_w, device=dev, dtype=dt),
            indexing="ij",
        )
        anchors = torch.stack([grid_x, grid_y], dim=-1) * self.space
        origin_coord = (
            torch.tensor(
                [w % self.space or self.space, h % self.space or self.space],
                device=dev,
                dtype=dt,
            )
            / 2.0
        )
        anchors = anchors + origin_coord
        return anchors.unsqueeze(0).repeat(bs, 1, 1, 1)

    def forward(self, features, images):
        target_size = features[0].shape[2:]

        # 1. 多尺度金字塔融合
        fused_feat = self.fusion(features)
        
        # 3. 各向异性切比雪夫图神经网络演化
        x_gnn = self.gnn1(fused_feat)
        x_gnn = self.gnn2(x_gnn)
        
        #x_gnn = fused_feat
        # 4. 预测细胞核质心中心热力图 (Center Heatmap)
        center_mask = self.center_head(x_gnn)

        # 5. 坐标与分类预测，利用中心热力图作为 Spatial Weight 增强回归精度
        anchors = self._generate_anchors(images, target_size[0], target_size[1])
        #reg_out = self.reg_head(x_gnn * (1.0 + center_mask)).permute(0, 2, 3, 1).contiguous()
        reg_out = self.reg_head(x_gnn).permute(0, 2, 3, 1).contiguous()
        cls_out = self.cls_head(x_gnn).permute(0, 2, 3, 1).contiguous()

        reg_offset = (torch.sigmoid(reg_out) - 0.5) * (2 * self.space)
        pred_coords = anchors + reg_offset

        return pred_coords, cls_out


# =====================================================================
# 4. 完整入口网络 DeNuCDet (Pathology Optimized)
# =====================================================================
class DeNuCDet(nn.Module):

    def __init__(self, backbone_cfg=None, fpn_cfg=None, head_cfg=None):
        super().__init__()

        fpn_cfg = fpn_cfg or {}
        head_cfg = head_cfg or {}

        in_channels = fpn_cfg.get("in_channels", [48, 96, 192])
        feat_channels = head_cfg.get("feat_channels", 96)
        num_classes = head_cfg.get("num_classes", 2)
        space = head_cfg.get("space", 8)

        self.backbone = build_backbone(backbone_cfg)

        self.point_gnn_detector = PathologyNucleiGNNDetector(
            in_channels_list=in_channels,
            feat_channels=feat_channels,
            num_classes=num_classes,
            space=space,
        )

    def forward(self, x):
        features = self.backbone(x)
        pred_coords, pred_logits = self.point_gnn_detector(features, x)

        return {
            "pred_coords": pred_coords.flatten(1, 2).contiguous(),
            "pred_logits": pred_logits.flatten(1, 2).contiguous()
        }




# =========================
# Model builders (presets)
# =========================
def denuc_det_shufflenet_x0_5():
    backbone_cfg = {
        "name": "ShuffleNetV2",
        "model_size": "0.5x",
        "out_stages": [2, 3, 4],
        "activation": "LeakyReLU",
        "pretrain": True,
    }
    fpn_cfg = {
        "name": "PAN",
        "in_channels": [48, 96, 192],
        "out_channels": 96,
    }
    head_cfg = {
        "name": "DeNuCHead",
        "in_channels": 96,
        "num_in": 3,
        "num_classes": 2,
        "feat_channels": 96,
    }
    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_shufflenet_x1_0():
    backbone_cfg = {
        "name": "ShuffleNetV2",
        "model_size": "1.0x",
        "out_stages": [2, 3, 4],
        "activation": "LeakyReLU",
        "pretrain": True,
    }
    fpn_cfg = {
        "name": "PAN",
        "in_channels": [116, 232, 464],
        "out_channels": 96,
    }
    head_cfg = {
        "name": "DeNuCHead",
        "in_channels": 96,
        "num_in": 3,
        "num_classes": 2,
        "feat_channels": 96,
    }
    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_shufflenet_x1_5():
    backbone_cfg = {
        "name": "ShuffleNetV2",
        "model_size": "1.5x",
        "out_stages": [2, 3, 4],
        "activation": "LeakyReLU",
        "pretrain": True,
    }
    fpn_cfg = {
        "name": "PAN",
        "in_channels": [176, 352, 704],
        "out_channels": 192,
    }
    head_cfg = {
        "name": "DeNuCHead",
        "in_channels": 192,
        "num_in": 3,
        "num_classes": 2,
        "feat_channels": 192,
    }
    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_shufflenet_x2_0():
    backbone_cfg = {
        "name": "ShuffleNetV2",
        "model_size": "2.0x",
        "out_stages": [2, 3, 4],
        "activation": "LeakyReLU",
        "pretrain": True,
    }
    fpn_cfg = {
        "name": "PAN",
        "in_channels": [244, 488, 976],
        "out_channels": 192,
    }
    head_cfg = {
        "name": "DeNuCHead",
        "in_channels": 192,
        "num_in": 3,
        "num_classes": 2,
        "feat_channels": 192,
    }
    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


def denuc_det_r50():
    backbone_cfg = {
        "name": "ResNet50",
        "out_stages": [2, 3, 4],
        "pretrain": True,
    }
    fpn_cfg = {
        "name": "PAN",
        "in_channels": [512, 1024, 2048],
        "out_channels": 256,
    }
    head_cfg = {
        "name": "DeNuCHead",
        "in_channels": 256,
        "num_in": 3,
        "num_classes": 2,
        "feat_channels": 256,
    }
    return DeNuCDet(backbone_cfg, fpn_cfg, head_cfg)


__all__ = {
    "denuc_det_shufflenet_x0_5": denuc_det_shufflenet_x0_5,
    "denuc_det_shufflenet_x1_0": denuc_det_shufflenet_x1_0,
    "denuc_det_shufflenet_x1_5": denuc_det_shufflenet_x1_5,
    "denuc_det_shufflenet_x2_0": denuc_det_shufflenet_x2_0,
    "denuc_det_r50": denuc_det_r50
}







