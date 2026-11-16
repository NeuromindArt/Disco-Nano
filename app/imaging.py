"""Pictures and masks between PIL, numpy and torch, the few things done to masks everywhere, growing and blurring, and
start images turned upright and fitted to the frame.

A picture is a 1x3xHxW tensor of 0 to 1, or of -1 to 1 where `signed` says so, as diffusion models and their
autoencoders work in. A mask is a 1x1xHxW tensor of 0 to 1, or a PIL "L" image of 0 to 255.

torch loads only when a function needs it, so the server can use the rest without it.
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def to_pil(picture, signed=False):
    """A picture tensor as an RGB PIL image."""
    if signed:
        picture = picture.add(1).div(2)
    return Image.fromarray((picture[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


def from_pil(image, like=None, device="cpu", signed=False):
    """An RGB PIL image as a picture tensor, on `like`'s device and dtype when one is given."""
    import torch
    picture = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    if signed:
        picture = picture * 2 - 1
    return picture.to(like.device, like.dtype) if like is not None else picture.to(device)


def mask_to_pil(mask):
    """A mask tensor as a PIL "L" image."""
    return Image.fromarray((mask[0, 0].float().clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


def mask_from_pil(image, device="cpu"):
    """A PIL image as a mask tensor, from its brightness."""
    import torch
    return torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32) / 255)[None, None].to(device)


def grow(mask, by):
    """A numpy mask grown `by` pixels each way, as a square: the most of each row's, then each column's, neighbors."""
    if by <= 0:
        return mask
    for axis in (0, 1):
        padded = np.pad(mask, [(by, by) if a == axis else (0, 0) for a in (0, 1)])
        mask = np.lib.stride_tricks.sliding_window_view(padded, by * 2 + 1, axis=axis).max(-1)
    return mask


def blur(x, sigma):
    """A Gaussian blur of a BxCxHxW tensor, across and then down, with the edges carried on outward."""
    import torch
    import torch.nn.functional as F
    if sigma <= 0:
        return x
    radius = max(int(3 * sigma), 1)
    kernel = torch.exp(-torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = x.shape[1]
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)


def upright(path):
    """An image as it is meant to be seen. Phone photos store their rotation in a tag."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def size_start_image(image, width, height, mode):
    """Fill stretches to the frame. Crop cuts the image to the frame's shape. Fit keeps the whole
    image over a blurred copy of itself, because the renderer would paint flat bars as flat bars."""
    size = (width, height)
    if mode == "fill":
        return image.resize(size, Image.LANCZOS)
    cropped = ImageOps.fit(image, size, Image.LANCZOS)
    if mode != "fit":
        return cropped
    canvas = cropped.filter(ImageFilter.GaussianBlur(max(size) / 24))
    inner = ImageOps.contain(image, size, Image.LANCZOS)
    canvas.paste(inner, ((width - inner.width) // 2, (height - inner.height) // 2))
    return canvas
