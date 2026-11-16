"""What steers each step toward the prompt: the cutouts of the picture that CLIP compares with it, and the losses.

Derived from Disco Diffusion, which is MIT licensed. See CREDITS.md and LICENSES/disco-diffusion.txt for its
authors. The cutouts and the losses are theirs; sizing LPIPS to the card is Nano's.
"""
import math

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from resize_right import resize
from torch import nn
from torch.nn import functional as F

import memory


normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


class CpuRotation:
    """PyTorch's GPU backward for this transform is not repeatable, so Exact repeat runs it on the CPU."""

    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(self, cuts):
        return self.rotation(cuts.cpu()).to(cuts.device)


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size, overview, inner_crop, ic_size_pow, ic_grey_p, skip_augs, exact_repeat=False, evidence=None,
                 extra=()):
        super().__init__()
        self.evidence, self.boxes, self.extra = evidence, [], list(extra)  # extra: (left, top, size) close-ups, such as faces
        self.cut_size, self.overview, self.inner_crop = cut_size, overview, inner_crop
        self.ic_size_pow, self.ic_grey_p, self.skip_augs = ic_size_pow, ic_grey_p, skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), interpolation=T.InterpolationMode.BILINEAR),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ])
        if exact_repeat:
            self.augs.transforms[2] = CpuRotation(self.augs.transforms[2])

    def forward(self, input, whole=None):
        """Whole-image cutouts look at `whole` when it is given, and close-ups always look at `input`."""
        cutouts = []
        gray = T.Grayscale(3)
        side_y, side_x = input.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        seen = input if whole is None else whole
        seen_y, seen_x = seen.shape[2:4]
        seen_min = min(seen_x, seen_y)
        pad_input = F.pad(seen, ((seen_y - seen_min) // 2, (seen_y - seen_min) // 2,
                                 (seen_x - seen_min) // 2, (seen_x - seen_min) // 2))
        cutout = resize(pad_input, out_shape=output_shape)
        if 0 < self.overview <= 4:
            cutouts.append(cutout)
            if self.overview >= 2:
                cutouts.append(gray(cutout))
            if self.overview >= 3:
                cutouts.append(TF.hflip(cutout))
            if self.overview == 4:
                cutouts.append(gray(TF.hflip(cutout)))
        elif self.overview > 4:
            cutouts += [cutout] * self.overview
        for i in range(self.inner_crop):
            size = int(torch.rand([]) ** self.ic_size_pow * (max_size - min_size) + min_size)
            spot = self.evidence.pick(size, side_x, side_y) if self.evidence else None
            if spot:
                offset_x, offset_y = spot
            else:
                offset_x = torch.randint(0, side_x - size + 1, ())
                offset_y = torch.randint(0, side_y - size + 1, ())
            self.boxes.append((int(offset_x), int(offset_y), size))  # where each close-up is, for evidence and regions
            cutout = input[:, :, offset_y:offset_y + size, offset_x:offset_x + size]
            if i <= int(self.ic_grey_p * self.inner_crop):
                cutout = gray(cutout)
            cutouts.append(resize(cutout, out_shape=output_shape))
        for offset_x, offset_y, size in self.extra:
            self.boxes.append((offset_x, offset_y, size))
            cutouts.append(resize(input[:, :, offset_y:offset_y + size, offset_x:offset_x + size], out_shape=output_shape))
        cutouts = torch.cat(cutouts)
        return cutouts if self.skip_augs else self.augs(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


# LPIPS runs VGG on the whole frame with gradients, about 3.9KB per pixel: 3.6GB at 1280x768, which an 8GB card
# doesn't have to spare. There, frames larger than 768x512 are compared at 768x512, which was tested at every
# size and still shows the start image's layout and colors that LPIPS guards. A larger card compares at the
# largest size that fits in what is free at the first step, full size when it can.
LPIPS_FLOOR_PIXELS = 768 * 512
LPIPS_BYTES_PER_PIXEL = 3930  # what its forward and backward pass add per compared pixel, measured from 512x320 to 1280x768
LPIPS_FIXED_UP_TO_GB = 8.5  # cards this size or smaller keep the tested 768x512
_lpips_pixels = {}


def forget_lpips_sizes():
    """A new render: the sizes LPIPS compared at are worked out again, as it may load other models."""
    _lpips_pixels.clear()


def lpips_pixels(pixels):
    """How many pixels LPIPS compares at, worked out once per render: the next render may load other models."""
    if pixels not in _lpips_pixels:
        if torch.cuda.get_device_properties(0).total_memory <= LPIPS_FIXED_UP_TO_GB * 2**30:
            _lpips_pixels[pixels] = min(pixels, LPIPS_FLOOR_PIXELS)
            return _lpips_pixels[pixels]
        free = (memory.cap_bytes or torch.cuda.mem_get_info()[0]) - torch.cuda.memory_allocated()
        fits = int(free * 0.75 / LPIPS_BYTES_PER_PIXEL)  # slack for the allocator's rounding and later steps
        _lpips_pixels[pixels] = min(pixels, max(fits, LPIPS_FLOOR_PIXELS))
    return _lpips_pixels[pixels]


def lpips_distance(model, image, target):
    pixels = image.shape[-2] * image.shape[-1]
    allowed = lpips_pixels(pixels)
    if pixels > allowed:
        scale = math.sqrt(allowed / pixels)
        size = (max(round(image.shape[-2] * scale), 32), max(round(image.shape[-1] * scale), 32))
        image = F.interpolate(image, size=size, mode="area")
        target = F.interpolate(target, size=size, mode="area")
    return model(image, target)


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])
