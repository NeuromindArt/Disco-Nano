"""In-between frames with RIFE 4.7, or optical flow when RIFE isn't downloaded.

The network layout is RIFE's IFNet (HDv3, version 4.7) from Practical-RIFE by Zhewei Huang (hzwer),
MIT license, as packaged by HolyWu's vs-rife, also MIT. The weights are the same as Practical-RIFE's
4.7 release. Only what inference needs is kept here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import imaging
import models


def warp(image, flow, grid):
    """Reads image at each pixel moved by flow, which is in pixels."""
    height, width = image.shape[-2:]
    flow = torch.cat([flow[:, :1] / ((width - 1) / 2), flow[:, 1:2] / ((height - 1) / 2)], 1)
    return F.grid_sample(image, (grid + flow).permute(0, 2, 3, 1), mode="bilinear", padding_mode="border",
                         align_corners=True)


def conv(c_in, c_out, stride):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride, 1), nn.LeakyReLU(0.2, True))


class ResConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, c_in, c):
        super().__init__()
        self.conv0 = nn.Sequential(conv(c_in, c // 2, 2), conv(c // 2, c, 2))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x, flow, scale):
        x = F.interpolate(x, scale_factor=1 / scale, mode="bilinear")
        if flow is not None:
            x = torch.cat((x, F.interpolate(flow, scale_factor=1 / scale, mode="bilinear") / scale), 1)
        out = F.interpolate(self.lastconv(self.convblock(self.conv0(x))), scale_factor=scale, mode="bilinear")
        return out[:, :4] * scale, out[:, 4:5]


class IFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, 192)
        self.block1 = IFBlock(8 + 4 + 8, 128)
        self.block2 = IFBlock(8 + 4 + 8, 96)
        self.block3 = IFBlock(8 + 4 + 8, 64)
        self.encode = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ConvTranspose2d(16, 4, 4, 2, 1))

    def forward(self, img0, img1, t):
        """Two frames, 0 to 1, padded to a multiple of 32. t is how far between them, 0 to 1."""
        height, width = img0.shape[-2:]
        ys = torch.linspace(-1, 1, height, device=img0.device).view(1, 1, height, 1).expand(1, 1, height, width)
        xs = torch.linspace(-1, 1, width, device=img0.device).view(1, 1, 1, width).expand(1, 1, height, width)
        grid = torch.cat([xs, ys], 1)
        f0, f1 = self.encode(img0), self.encode(img1)
        timestep = torch.full_like(img0[:, :1], t)
        flow = mask = None
        warped0, warped1 = img0, img1
        for block, scale in zip((self.block0, self.block1, self.block2, self.block3), (8, 4, 2, 1)):
            if flow is None:
                flow, mask = block(torch.cat((img0, img1, f0, f1, timestep), 1), None, scale)
            else:
                wf0, wf1 = warp(f0, flow[:, :2], grid), warp(f1, flow[:, 2:4], grid)
                change, mask = block(torch.cat((warped0, warped1, wf0, wf1, timestep, mask), 1), flow, scale)
                flow = flow + change
            warped0, warped1 = warp(img0, flow[:, :2], grid), warp(img1, flow[:, 2:4], grid)
        mask = torch.sigmoid(mask)
        return warped0 * mask + warped1 * (1 - mask)


def load(device):
    """RIFE on the device, or None when it isn't downloaded."""
    import os
    path = models.path_of(models.RIFE)
    if not os.path.exists(path) or os.environ.get("NANO_NO_RIFE"):  # set it to try the fallback
        return None
    weights = torch.load(path, map_location="cpu", weights_only=True)
    net = IFNet()
    net.load_state_dict({k.replace("module.", ""): v for k, v in weights.items()})
    return net.eval().to(device)


@torch.no_grad()
def between_rife(net, a, b, count):
    """count frames evenly between a and b, which are -1 to 1."""
    height, width = a.shape[-2:]
    pad = (0, -width % 32, 0, -height % 32)
    a0, b0 = (F.pad(x.add(1).div(2).float(), pad, mode="replicate") for x in (a, b))
    return [net(a0, b0, (k + 1) / (count + 1))[..., :height, :width].clamp(0, 1).mul(2).sub(1) for k in range(count)]


def between_flow(a, b, count):
    """The fallback: each frame is moved part of the way along the optical flow both ways, then mixed."""
    import animate
    from PIL import Image
    cfg = {"sampling_mode": "bilinear", "padding_mode": "border"}
    pictures = [imaging.to_pil(x, signed=True) for x in (a, b)]
    back = animate.video_flow(pictures[0], pictures[1]).to(a.device)  # each pixel of b, where it was in a
    ahead = animate.video_flow(pictures[1], pictures[0]).to(a.device)  # each pixel of a, where it went in b
    ys, xs = animate.pixel_grid(*a.shape[-2:], a.device)
    out = []
    for k in range(count):
        t = (k + 1) / (count + 1)
        from_a = animate.sample(a, xs + t * back[..., 0], ys + t * back[..., 1], cfg)
        from_b = animate.sample(b, xs + (1 - t) * ahead[..., 0], ys + (1 - t) * ahead[..., 1], cfg)
        out.append((1 - t) * from_a + t * from_b)
    return out


class Smoother:
    """Makes in-between frames, with RIFE when it's on disk."""

    def __init__(self, device, status):
        self.net = load(device)
        status("Adding in-between frames" + ("" if self.net else " with optical flow (get RIFE from Models for smoother ones)"))

    def between(self, a, b, count):
        return between_rife(self.net, a, b, count) if self.net is not None else between_flow(a, b, count)
