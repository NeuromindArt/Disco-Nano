"""Measures how near each part of a picture looks, in a way that gradients can pass through."""
import os

import torch
import torch.nn.functional as F
from resize_right import resize

import models
import paths

PATCH = 14
LONG_SIDE = 392  # enough to read the layout, and the render needs a fresh pass at every step
MEAN = (0.485, 0.456, 0.406)
SPREAD = (0.229, 0.224, 0.225)


class Depth:
    def __init__(self, device, status):
        models.ensure(models.DEPTH, status)
        try:
            from transformers import DepthAnythingForDepthEstimation
        except ImportError as e:
            raise ValueError("Depth target needs the transformers package. Run setup again to add it.") from e
        folder = os.path.join(paths.MODELS, models.DEPTH)
        self.net = DepthAnythingForDepthEstimation.from_pretrained(folder).to(device).eval().requires_grad_(False)
        self.mean = torch.tensor(MEAN, device=device)[None, :, None, None]
        self.spread = torch.tensor(SPREAD, device=device)[None, :, None, None]

    @torch.no_grad()
    def closeness(self, image):
        """Depth Anything's relative closeness for an image of 0 to 1, at the image's own size. Higher is nearer."""
        height, width = image.shape[-2:]
        shrink = 518 / max(height, width)  # a finer read, since a camera move shows every error
        shape = [max(round(side * shrink / PATCH), 1) * PATCH for side in (height, width)]
        sized = resize(image.float(), out_shape=[image.shape[0], 3, *shape])
        near = self.net((sized - self.mean) / self.spread).predicted_depth[:, None]
        return F.interpolate(near, size=(height, width), mode="bilinear", align_corners=False)[:, 0]

    def relief(self, image):
        """Depth from an image of 0 to 1, with its scale and offset removed, since only its shape counts."""
        height, width = image.shape[-2:]
        shrink = LONG_SIDE / max(height, width)
        shape = [max(round(side * shrink / PATCH), 1) * PATCH for side in (height, width)]
        sized = resize(image.float(), out_shape=[image.shape[0], 3, *shape])
        depth = self.net((sized - self.mean) / self.spread).predicted_depth
        centered = depth - depth.mean((1, 2), keepdim=True)
        return centered / centered.abs().mean((1, 2), keepdim=True).clamp(min=1e-6)
