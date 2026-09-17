from __future__ import annotations

from types import MappingProxyType

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNReLU(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        padding: int = 1,
        dilation: int = 1,
    ):
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )


class HyperComplexConv2d(nn.Module):
    """Quaternion-style grouped convolution used by medium/large ASPP branches."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        if in_channels % 4 != 0 or out_channels % 4 != 0:
            raise ValueError(
                f"HyperComplexConv2d needs channels divisible by 4: in={in_channels}, out={out_channels}"
            )
        self.in_g = in_channels // 4
        self.out_g = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.r = nn.Parameter(
            torch.empty(self.out_g, self.in_g, kernel_size, kernel_size)
        )
        self.i = nn.Parameter(
            torch.empty(self.out_g, self.in_g, kernel_size, kernel_size)
        )
        self.j = nn.Parameter(
            torch.empty(self.out_g, self.in_g, kernel_size, kernel_size)
        )
        self.k = nn.Parameter(
            torch.empty(self.out_g, self.in_g, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for weight in (self.r, self.i, self.j, self.k):
            nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def conv(self, x, w):
        return F.conv2d(x, w, None, self.stride, self.padding, self.dilation)

    def forward(self, x):
        x0, x1, x2, x3 = torch.chunk(x, 4, dim=1)
        r0, r1, r2, r3 = (
            self.conv(x0, self.r),
            self.conv(x1, self.r),
            self.conv(x2, self.r),
            self.conv(x3, self.r),
        )
        i0, i1, i2, i3 = (
            self.conv(x0, self.i),
            self.conv(x1, self.i),
            self.conv(x2, self.i),
            self.conv(x3, self.i),
        )
        j0, j1, j2, j3 = (
            self.conv(x0, self.j),
            self.conv(x1, self.j),
            self.conv(x2, self.j),
            self.conv(x3, self.j),
        )
        k0, k1, k2, k3 = (
            self.conv(x0, self.k),
            self.conv(x1, self.k),
            self.conv(x2, self.k),
            self.conv(x3, self.k),
        )
        y = torch.cat(
            [
                r0 - i1 - j2 - k3,
                i0 + r1 + k2 - j3,
                j0 - k1 + r2 + i3,
                k0 + j1 - i2 + r3,
            ],
            dim=1,
        )
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class HConvGNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__(
            HyperComplexConv2d(
                in_ch,
                out_ch,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )


class PHMConv2d(nn.Module):
    """Parameterized Hypercomplex Multiplication convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=2,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super().__init__()
        if in_channels % n != 0 or out_channels % n != 0:
            raise ValueError(
                f"PHMConv2d needs in/out divisible by n: in={in_channels}, out={out_channels}, n={n}"
            )
        self.n = int(n)
        self.in_g = in_channels // self.n
        self.out_g = out_channels // self.n
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.A = nn.Parameter(torch.empty(self.n, self.n, self.n))
        self.S = nn.Parameter(
            torch.empty(self.n, self.out_g, self.in_g, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=5**0.5)
        for r in range(self.n):
            nn.init.kaiming_normal_(self.S[r], mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        xs = torch.chunk(x, self.n, dim=1)
        ys = []
        for i in range(self.n):
            yi = None
            for j in range(self.n):
                w_ij = torch.sum(
                    self.A[:, i, j].view(self.n, 1, 1, 1, 1) * self.S, dim=0
                )
                yij = F.conv2d(
                    xs[j], w_ij, None, self.stride, self.padding, self.dilation
                )
                yi = yij if yi is None else yi + yij
            ys.append(yi)
        y = torch.cat(ys, dim=1)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y


class PHMConvGNReLU(nn.Sequential):
    def __init__(
        self, in_ch, out_ch, n=2, kernel_size=1, padding=0, dilation=1, dropout=0.1
    ):
        super().__init__(
            PHMConv2d(
                in_ch,
                out_ch,
                n=n,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )


class AvgPoolBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return F.interpolate(
            self.net(x), size=x.shape[-2:], mode="bilinear", align_corners=False
        )


class AvgMaxPoolBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return F.interpolate(
            self.proj(torch.cat([avg, mx], dim=1)),
            size=size,
            mode="bilinear",
            align_corners=False,
        )


class StripPoolBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.pre(x)
        h, w = z.shape[-2:]
        yh = z.mean(dim=3, keepdim=True).expand(-1, -1, h, w)
        yw = z.mean(dim=2, keepdim=True).expand(-1, -1, h, w)
        return self.fuse(torch.cat([yh, yw], dim=1))


# Coordinate-attention structural reference: houqb/CoordAttention (MIT).
# Reference copyright (c) 2021 Qibin (Andrew) Hou.
# See licenses/CoordAttention.txt and THIRD_PARTY_NOTICES.md.
class CoordBranch(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=32):
        super().__init__()
        mid = max(16, out_ch // reduction)
        self.pre = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )
        self.shared = nn.Sequential(
            nn.Conv2d(out_ch, mid, 1, bias=False),
            nn.GroupNorm(min(16, mid), mid),
            nn.ReLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(mid, out_ch, 1)
        self.conv_w = nn.Conv2d(mid, out_ch, 1)
        self.out = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        z = self.pre(x)
        _, _, h, w = z.shape
        fh = z.mean(dim=3, keepdim=True)
        fw = z.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        f = self.shared(torch.cat([fh, fw], dim=2))
        fh, fw = torch.split(f, [h, w], dim=2)
        fw = fw.permute(0, 1, 3, 2)
        return self.out(
            z * torch.sigmoid(self.conv_h(fh)) * torch.sigmoid(self.conv_w(fw))
        )


def make_global_branch(name: str, in_ch: int, out_ch: int):
    if name == "avg":
        return AvgPoolBranch(in_ch, out_ch)
    if name == "avgmax":
        return AvgMaxPoolBranch(in_ch, out_ch)
    if name == "strip":
        return StripPoolBranch(in_ch, out_ch)
    if name == "coord":
        return CoordBranch(in_ch, out_ch)
    raise ValueError(f"unknown global_branch: {name}")


class SKFusion(nn.Module):
    def __init__(
        self,
        channels: int,
        num_branches: int = 4,
        reduction: int = 16,
        min_dim: int = 32,
    ):
        super().__init__()
        hidden = max(min_dim, channels // reduction)
        self.num_branches = num_branches
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels * num_branches, 1, bias=True),
        )

    def forward(self, feats):
        u = torch.stack(feats, dim=1).sum(dim=1)
        logits = self.fc(u)
        b, cm, _, _ = logits.shape
        c = cm // self.num_branches
        weights = torch.softmax(logits.view(b, self.num_branches, c, 1, 1), dim=1)
        return (torch.stack(feats, dim=1) * weights).sum(dim=1), weights


class PHMProjectLocalSKGlobalBypassHComplexASPP(nn.Module):
    def __init__(
        self,
        in_ch=512,
        out_ch=256,
        rates=(12, 24, 36),
        global_branch="avg",
        reduction=16,
        dropout=0.1,
        phm_n=2,
    ):
        super().__init__()
        if len(rates) != 3:
            raise ValueError("rates must contain exactly three dilation rates")
        r1, r2, r3 = rates
        self.b1 = ConvGNReLU(in_ch, out_ch, 1, 0, 1)
        self.b_small = ConvGNReLU(in_ch, out_ch, 3, r1, r1)
        self.b_mid = HConvGNReLU(in_ch, out_ch, 3, r2, r2)
        self.b_large = HConvGNReLU(in_ch, out_ch, 3, r3, r3)
        self.b_global = make_global_branch(global_branch, in_ch, out_ch)
        self.sk = SKFusion(out_ch, num_branches=4, reduction=reduction)
        self.project = PHMConvGNReLU(
            out_ch * 2, out_ch, n=phm_n, kernel_size=1, padding=0, dropout=dropout
        )

    def forward(self, x):
        y_local, _ = self.sk(
            [self.b1(x), self.b_small(x), self.b_mid(x), self.b_large(x)]
        )
        y_global = self.b_global(x)
        return self.project(torch.cat([y_local, y_global], dim=1))


class LocalSKGlobalBypassDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        global_branch="avg",
        encoder_output_stride=8,
        aspp_out_ch=256,
        decoder_ch=256,
        low_ch=48,
        rates=(12, 24, 36),
        sk_reduction=16,
        phm_n=2,
        encoder_weights="imagenet",
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            "resnet18",
            in_channels=3,
            depth=5,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        channels = self.encoder.out_channels
        self.low_idx = 2
        self.high_idx = -1
        self.aspp = PHMProjectLocalSKGlobalBypassHComplexASPP(
            in_ch=channels[self.high_idx],
            out_ch=aspp_out_ch,
            rates=rates,
            global_branch=global_branch,
            reduction=sk_reduction,
            phm_n=phm_n,
        )
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels[self.low_idx], low_ch, 1, bias=False),
            nn.GroupNorm(min(16, low_ch), low_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvGNReLU(aspp_out_ch + low_ch, decoder_ch, 3, 1),
            ConvGNReLU(decoder_ch, decoder_ch, 3, 1),
        )
        self.head = nn.Conv2d(decoder_ch, 1, 1)

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        low, high = feats[self.low_idx], feats[self.high_idx]
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, self.low_proj(low)], dim=1)
        y = self.head(self.decoder(y))
        return F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
#  Ablation ASPP & Model  (A2-Conv / A2-HC / A3)
# ---------------------------------------------------------------------------


class AblationASPP(nn.Module):
    """LocalSK-based ASPP for ablation studies.

    Configurable via:
      - branch_type:       "conv" | "hypercomplex"
      - use_global_branch: True | False
      - projection_type:   "none" | "conv" | "phm"
      - phm_n:             PHM order (only used for projection_type="phm")

    Mapping to ablation variants:
      A2-Conv:  branch_type="conv",  use_global_branch=False, projection_type="none"
      A2-HC:    branch_type="hypercomplex", use_global_branch=False, projection_type="none"
      A3:       branch_type="hypercomplex", use_global_branch=True,  projection_type="conv"
      A4 (orig):branch_type="hypercomplex", use_global_branch=True,  projection_type="phm", phm_n=2
    """

    def __init__(
        self,
        in_ch: int = 512,
        out_ch: int = 256,
        rates: tuple = (12, 24, 36),
        branch_type: str = "hypercomplex",
        use_global_branch: bool = True,
        global_branch: str = "avg",
        reduction: int = 16,
        projection_type: str = "phm",
        phm_n: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(rates) != 3:
            raise ValueError("rates must contain exactly three dilation rates")
        r1, r2, r3 = rates

        # ---- local multi-scale branches ----
        self.b1 = ConvGNReLU(in_ch, out_ch, 1, 0, 1)  # 1×1
        self.b_small = ConvGNReLU(in_ch, out_ch, 3, r1, r1)  # 3×3, dil=12

        if branch_type == "conv":
            self.b_mid = ConvGNReLU(in_ch, out_ch, 3, r2, r2)  # 3×3, dil=24
            self.b_large = ConvGNReLU(in_ch, out_ch, 3, r3, r3)  # 3×3, dil=36
        elif branch_type == "hypercomplex":
            self.b_mid = HConvGNReLU(in_ch, out_ch, 3, r2, r2)
            self.b_large = HConvGNReLU(in_ch, out_ch, 3, r3, r3)
        else:
            raise ValueError(f"unknown branch_type: {branch_type}")

        self.branch_type = branch_type
        self._rates = list(rates)

        # ---- LocalSK fusion (always 4 branches) ----
        self.sk = SKFusion(out_ch, num_branches=4, reduction=reduction)

        # ---- global context branch ----
        self.use_global_branch = use_global_branch
        if use_global_branch:
            self.b_global = make_global_branch(global_branch, in_ch, out_ch)

        # ---- projection after local+global concat ----
        self.projection_type = projection_type
        self.phm_n = phm_n if projection_type == "phm" else None

        if use_global_branch and projection_type != "none":
            proj_in = out_ch * 2  # 256 + 256 = 512
            if projection_type == "conv":
                self.project = nn.Sequential(
                    nn.Conv2d(proj_in, out_ch, 1, bias=False),
                    nn.GroupNorm(min(32, out_ch), out_ch),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                )
            elif projection_type == "phm":
                self.project = PHMConvGNReLU(
                    proj_in,
                    out_ch,
                    n=phm_n,
                    kernel_size=1,
                    padding=0,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"unknown projection_type: {projection_type}")

    def forward(self, x):
        y_local, _ = self.sk(
            [self.b1(x), self.b_small(x), self.b_mid(x), self.b_large(x)]
        )

        if not self.use_global_branch:
            # A2-Conv / A2-HC: LocalSK output → decoder
            return y_local

        y_global = self.b_global(x)
        y = torch.cat([y_local, y_global], dim=1)

        if self.projection_type != "none":
            return self.project(y)

        # Fallback: no projection (512 ch out – not used by any current decoder)
        return y


class AblationDeepLabV3Plus(nn.Module):
    """DeepLabV3+-style model with configurable LocalSK ASPP for ablation studies.

    Shares the same encoder, low-level projection, decoder, and head as the
    original ``LocalSKGlobalBypassDeepLabV3Plus`` so that comparisons are fair.
    """

    def __init__(
        self,
        branch_type: str = "hypercomplex",
        use_global_branch: bool = True,
        global_branch: str = "avg",
        projection_type: str = "phm",
        phm_n: int = 2,
        encoder_output_stride: int = 8,
        aspp_out_ch: int = 256,
        decoder_ch: int = 256,
        low_ch: int = 48,
        rates: tuple = (12, 24, 36),
        sk_reduction: int = 16,
        dropout: float = 0.1,
        encoder_weights: str = "imagenet",
    ):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.encoder = smp.encoders.get_encoder(
            "resnet18",
            in_channels=3,
            depth=5,
            weights=encoder_weights,
            output_stride=encoder_output_stride,
        )
        channels = self.encoder.out_channels
        self.low_idx = 2
        self.high_idx = -1

        self.aspp = AblationASPP(
            in_ch=channels[self.high_idx],
            out_ch=aspp_out_ch,
            rates=rates,
            branch_type=branch_type,
            use_global_branch=use_global_branch,
            global_branch=global_branch,
            reduction=sk_reduction,
            projection_type=projection_type,
            phm_n=phm_n,
            dropout=dropout,
        )

        self.low_proj = nn.Sequential(
            nn.Conv2d(channels[self.low_idx], low_ch, 1, bias=False),
            nn.GroupNorm(min(16, low_ch), low_ch),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvGNReLU(aspp_out_ch + low_ch, decoder_ch, 3, 1),
            ConvGNReLU(decoder_ch, decoder_ch, 3, 1),
        )
        self.head = nn.Conv2d(decoder_ch, 1, 1)

        # Store config for checkpoint serialization
        self._model_config = {
            "branch_type": branch_type,
            "use_global_branch": use_global_branch,
            "global_branch": global_branch,
            "projection_type": projection_type,
            "phm_n": phm_n if projection_type == "phm" else None,
            "rates": list(rates),
            "encoder_output_stride": encoder_output_stride,
            "aspp_out_ch": aspp_out_ch,
            "decoder_ch": decoder_ch,
            "low_ch": low_ch,
            "sk_reduction": sk_reduction,
            "dropout": dropout,
        }

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        low, high = feats[self.low_idx], feats[self.high_idx]
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, self.low_proj(low)], dim=1)
        y = self.head(self.decoder(y))
        return F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
#  Strict supplementary ablation models
# ---------------------------------------------------------------------------


class MatchedStandardASPP(nn.Module):
    """Matched custom A0: five ordinary ASPP branches and dense projection."""

    def __init__(self, in_ch=512, out_ch=256, rates=(12, 24, 36), dropout=0.1):
        super().__init__()
        if len(rates) != 3:
            raise ValueError("rates must contain exactly three dilation rates")
        r1, r2, r3 = rates
        self.b1 = ConvGNReLU(in_ch, out_ch, 1, 0, 1)
        self.b_small = ConvGNReLU(in_ch, out_ch, 3, r1, r1)
        self.b_mid = ConvGNReLU(in_ch, out_ch, 3, r2, r2)
        self.b_large = ConvGNReLU(in_ch, out_ch, 3, r3, r3)
        self.b_global = AvgPoolBranch(in_ch, out_ch)
        self.project_operator = nn.Conv2d(out_ch * 5, out_ch, 1, bias=False)
        self.project_post = nn.Sequential(
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        branches = [
            self.b1(x),
            self.b_small(x),
            self.b_mid(x),
            self.b_large(x),
            self.b_global(x),
        ]
        return self.project_post(self.project_operator(torch.cat(branches, dim=1)))


class StrictSupplementaryASPP(nn.Module):
    """HC local branches with controlled fusion/global/projector variants."""

    def __init__(
        self,
        in_ch=512,
        out_ch=256,
        rates=(12, 24, 36),
        local_fusion="localsk",
        global_mode="bypass",
        global_branch="avg",
        reduction=16,
        projection_type="conv",
        projection_groups=1,
        projection_rank=None,
        phm_n=2,
        dropout=0.1,
    ):
        super().__init__()
        if len(rates) != 3:
            raise ValueError("rates must contain exactly three dilation rates")
        if global_mode not in {"bypass", "in_sk"}:
            raise ValueError(f"unknown global_mode: {global_mode}")
        if local_fusion not in {"localsk", "concat_conv", "mean", "sum"}:
            raise ValueError(f"unknown local_fusion: {local_fusion}")

        r1, r2, r3 = rates
        self.b1 = ConvGNReLU(in_ch, out_ch, 1, 0, 1)
        self.b_small = ConvGNReLU(in_ch, out_ch, 3, r1, r1)
        self.b_mid = HConvGNReLU(in_ch, out_ch, 3, r2, r2)
        self.b_large = HConvGNReLU(in_ch, out_ch, 3, r3, r3)
        self.b_global = make_global_branch(global_branch, in_ch, out_ch)
        self.local_fusion = local_fusion
        self.global_mode = global_mode

        if global_mode == "in_sk":
            if local_fusion != "localsk":
                raise ValueError("global_mode='in_sk' requires local_fusion='localsk'")
            self.sk = SKFusion(out_ch, num_branches=5, reduction=reduction)
        elif local_fusion == "localsk":
            self.sk = SKFusion(out_ch, num_branches=4, reduction=reduction)
        elif local_fusion == "concat_conv":
            # Intentionally bare: no bias, normalization, activation, or dropout.
            self.local_concat = nn.Conv2d(out_ch * 4, out_ch, 1, bias=False)

        proj_in = out_ch * 2
        if projection_type == "conv":
            self.project_operator = nn.Conv2d(proj_in, out_ch, 1, bias=False)
        elif projection_type == "phm":
            if proj_in % phm_n != 0 or out_ch % phm_n != 0:
                raise ValueError(
                    f"PHM channels must be divisible by n: in={proj_in}, out={out_ch}, n={phm_n}"
                )
            self.project_operator = PHMConv2d(
                proj_in, out_ch, n=phm_n, kernel_size=1, bias=False
            )
        elif projection_type == "grouped":
            if proj_in % projection_groups != 0 or out_ch % projection_groups != 0:
                raise ValueError(
                    "grouped projector channels must be divisible by projection_groups"
                )
            self.project_operator = nn.Conv2d(
                proj_in, out_ch, 1, groups=projection_groups, bias=False
            )
        elif projection_type == "lowrank":
            if not projection_rank or projection_rank <= 0:
                raise ValueError("lowrank projector requires a positive projection_rank")
            self.project_operator = nn.Sequential(
                nn.Conv2d(proj_in, projection_rank, 1, bias=False),
                nn.Conv2d(projection_rank, out_ch, 1, bias=False),
            )
        else:
            raise ValueError(f"unknown projection_type: {projection_type}")

        # Identical post-projector regularization for every compared projector.
        self.project_post = nn.Sequential(
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.projection_type = projection_type
        self.phm_n = phm_n if projection_type == "phm" else None

    def _fuse_local(self, local_feats):
        if self.local_fusion == "localsk":
            return self.sk(local_feats)[0]
        if self.local_fusion == "concat_conv":
            return self.local_concat(torch.cat(local_feats, dim=1))
        stacked = torch.stack(local_feats, dim=1)
        if self.local_fusion == "mean":
            return stacked.mean(dim=1)
        return stacked.sum(dim=1)

    def forward(self, x):
        local_feats = [self.b1(x), self.b_small(x), self.b_mid(x), self.b_large(x)]
        global_feat = self.b_global(x)

        if self.global_mode == "in_sk":
            all_feats = local_feats + [global_feat]
            _, weights = self.sk(all_feats)
            stacked = torch.stack(all_feats, dim=1)
            local_side = (stacked[:, :4] * weights[:, :4]).sum(dim=1)
            global_side = stacked[:, 4] * weights[:, 4]
        else:
            local_side = self._fuse_local(local_feats)
            global_side = global_feat

        fused = torch.cat([local_side, global_side], dim=1)
        return self.project_post(self.project_operator(fused))


class StrictSupplementaryDeepLabV3Plus(nn.Module):
    """Shared outer network for every new strict supplementary configuration."""

    def __init__(self, config, encoder_weights="imagenet"):
        super().__init__()
        import segmentation_models_pytorch as smp

        cfg = dict(config)
        self.encoder = smp.encoders.get_encoder(
            cfg["encoder_name"],
            in_channels=3,
            depth=5,
            weights=encoder_weights,
            output_stride=cfg["encoder_output_stride"],
        )
        channels = self.encoder.out_channels
        self.low_idx = 2
        self.high_idx = -1

        if cfg["context_block"] == "matched_standard_aspp":
            self.aspp = MatchedStandardASPP(
                in_ch=channels[self.high_idx],
                out_ch=cfg["aspp_out_ch"],
                rates=tuple(cfg["rates"]),
                dropout=cfg["dropout"],
            )
        elif cfg["context_block"] == "hc_localsk_global":
            self.aspp = StrictSupplementaryASPP(
                in_ch=channels[self.high_idx],
                out_ch=cfg["aspp_out_ch"],
                rates=tuple(cfg["rates"]),
                local_fusion=cfg["local_fusion"],
                global_mode=cfg["global_mode"],
                global_branch=cfg["global_branch"],
                reduction=cfg["sk_reduction"],
                projection_type=cfg["projection_type"],
                projection_groups=cfg["projection_groups"],
                projection_rank=cfg["projection_rank"],
                phm_n=cfg["phm_n"] or 2,
                dropout=cfg["dropout"],
            )
        else:
            raise ValueError(f"unknown context_block: {cfg['context_block']}")

        self.low_proj = nn.Sequential(
            nn.Conv2d(channels[self.low_idx], cfg["low_ch"], 1, bias=False),
            nn.GroupNorm(min(16, cfg["low_ch"]), cfg["low_ch"]),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvGNReLU(
                cfg["aspp_out_ch"] + cfg["low_ch"], cfg["decoder_ch"], 3, 1
            ),
            ConvGNReLU(cfg["decoder_ch"], cfg["decoder_ch"], 3, 1),
        )
        self.head = nn.Conv2d(cfg["decoder_ch"], 1, 1)
        self._model_config = cfg

    def forward(self, x):
        input_size = x.shape[-2:]
        feats = self.encoder(x)
        low, high = feats[self.low_idx], feats[self.high_idx]
        y = self.aspp(high)
        y = F.interpolate(y, size=low.shape[-2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, self.low_proj(low)], dim=1)
        y = self.head(self.decoder(y))
        return F.interpolate(y, size=input_size, mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
#  Helpers for model-name ↔ ablation-variant mapping
# ---------------------------------------------------------------------------

_COMMON_CONFIG = {
    "encoder_name": "resnet18",
    "encoder_weights": "imagenet",
    "encoder_output_stride": 8,
    "aspp_out_ch": 256,
    "decoder_ch": 256,
    "low_ch": 48,
    "rates": (12, 24, 36),
    "sk_reduction": 16,
    "dropout": 0.1,
    "global_branch": "avg",
    "projection_groups": 1,
    "projection_rank": None,
    "phm_n": None,
}


def _registry_entry(**overrides):
    cfg = dict(_COMMON_CONFIG)
    cfg.update(overrides)
    return MappingProxyType(cfg)


# Complete immutable construction registry. Legacy A2/A3/A4 keys retain their
# original classes and state-dict names; strict supplementary keys use the new
# shared wrapper above.
_MODEL_REGISTRY = MappingProxyType(
    {
        "phm_project_n2": _registry_entry(
            cls="original",
            ablation_variant="A4",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="bypass",
            projection_type="phm",
            phm_n=2,
        ),
        "localsk_conv_resnet18_pretrained": _registry_entry(
            cls="ablation",
            ablation_variant="A2-Conv",
            context_block="legacy_ablation",
            branch_type="conv",
            use_global_branch=False,
            local_fusion="localsk",
            global_mode="none",
            projection_type="none",
        ),
        "localsk_hc_resnet18_pretrained": _registry_entry(
            cls="ablation",
            ablation_variant="A2-HC",
            context_block="legacy_ablation",
            branch_type="hypercomplex",
            use_global_branch=False,
            local_fusion="localsk",
            global_mode="none",
            projection_type="none",
        ),
        "localsk_hc_global_conv_resnet18_pretrained": _registry_entry(
            cls="ablation",
            ablation_variant="A3",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="bypass",
            projection_type="conv",
        ),
        "matched_a0_standard_aspp_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="matched-A0",
            context_block="matched_standard_aspp",
            branch_type="conv",
            use_global_branch=True,
            local_fusion="concat5",
            global_mode="standard_aspp_concat",
            projection_type="conv",
        ),
        "localsk_hc_global_insk_conv_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="global-in-SK",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="in_sk",
            projection_type="conv",
        ),
        "concat_hc_global_conv_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="concat-fusion",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="concat_conv",
            global_mode="bypass",
            projection_type="conv",
        ),
        "mean_hc_global_conv_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="mean-fusion",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="mean",
            global_mode="bypass",
            projection_type="conv",
        ),
        "sum_hc_global_conv_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="sum-fusion",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="sum",
            global_mode="bypass",
            projection_type="conv",
        ),
        "localsk_hc_global_group2_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="grouped-projector",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="bypass",
            projection_type="grouped",
            projection_groups=2,
        ),
        "localsk_hc_global_lowrank85_resnet18_pretrained": _registry_entry(
            cls="supplementary",
            ablation_variant="lowrank-projector",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="bypass",
            projection_type="lowrank",
            projection_rank=85,
        ),
        "phm_project_n1": _registry_entry(
            cls="supplementary",
            ablation_variant="PHM-n1",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="bypass",
            projection_type="phm",
            phm_n=1,
        ),
        "phm_project_n4": _registry_entry(
            cls="supplementary",
            ablation_variant="PHM-n4",
            context_block="hc_localsk_global",
            branch_type="hypercomplex",
            use_global_branch=True,
            local_fusion="localsk",
            global_mode="bypass",
            projection_type="phm",
            phm_n=4,
        ),
    }
)


def get_ablation_variant(
    branch_type: str, use_global_branch: bool, projection_type: str
) -> str:
    """Return a human-readable ablation variant label from config."""
    if branch_type == "conv" and not use_global_branch and projection_type == "none":
        return "A2-Conv"
    if (
        branch_type == "hypercomplex"
        and not use_global_branch
        and projection_type == "none"
    ):
        return "A2-HC"
    if (
        branch_type == "hypercomplex"
        and use_global_branch
        and projection_type == "conv"
    ):
        return "A3"
    if branch_type == "hypercomplex" and use_global_branch and projection_type == "phm":
        return "A4"
    return "unknown"
