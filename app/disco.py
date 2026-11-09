"""Disco Diffusion's render path, for still images and animation frames.

Derived from Disco Diffusion, which is MIT licensed. See CREDITS.md and LICENSES/disco-diffusion.txt
for its authors. Cutouts, the losses, guidance and the secondary model are theirs. Their animation,
video and notebook code is not included; animation here comes from animate.py, which is Nano's own.
renderer.py runs this in its own process.
"""
import gc
import json
import math
import os
import random
import re
from dataclasses import dataclass
from functools import partial

import clip
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from PIL import Image, ImageFilter, ImageOps
from resize_right import resize
from torch import nn
from torch.nn import functional as F

import animate
import depth
import graft
import keyframes
import lab
import memory
import models
import partner
import paths
import rules
import tiles
from settings import DEFAULTS, uses_partner

device = torch.device("cuda")
normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
CLIP_NAMES = {"ViTB32": "ViT-B/32", "ViTB16": "ViT-B/16", "ViTL14": "ViT-L/14", "RN50": "RN50", "RN101": "RN101",
              "RN50x4": "RN50x4", "RN50x16": "RN50x16", "RN50x64": "RN50x64"}


def parse_prompt(prompt):
    text, _, weight = prompt.rpartition(":")
    try:
        return (text, float(weight)) if text else (prompt, 1.0)
    except ValueError:
        return prompt, 1.0


def schedule(text):
    """Reads a schedule such as [12]*400+[4]*600 into a list of 1000 values."""
    values = []
    for part in str(text).replace(" ", "").split("+"):
        match = re.fullmatch(r"\[(-?\d+(?:\.\d+)?)\]\*(\d+)", part)
        if not match:
            raise ValueError(f"Could not read the schedule {text!r}. Use a form like [12]*400+[4]*600.")
        values += [float(match.group(1))] * int(match.group(2))
    return (values + values[-1:] * 1000)[:1000]


class CpuRotation:
    """PyTorch's GPU backward for this transform is not repeatable, so Exact repeat runs it on the CPU."""

    def __init__(self, rotation):
        self.rotation = rotation

    def __call__(self, cuts):
        return self.rotation(cuts.cpu()).to(cuts.device)


class MakeCutoutsDango(nn.Module):
    def __init__(self, cut_size, overview, inner_crop, ic_size_pow, ic_grey_p, skip_augs, exact_repeat=False, evidence=None):
        super().__init__()
        self.evidence, self.boxes = evidence, []
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
            if self.evidence:
                self.boxes.append((int(offset_x), int(offset_y), size))
            cutout = input[:, :, offset_y:offset_y + size, offset_x:offset_x + size]
            if i <= int(self.ic_grey_p * self.inner_crop):
                cutout = gray(cutout)
            cutouts.append(resize(cutout, out_shape=output_shape))
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


def lpips_pixels(pixels):
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


# ---- secondary model ----
def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True))


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]
        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock([
                self.down,
                ConvBlock(cs[0], cs[1]),
                ConvBlock(cs[1], cs[1]),
                SkipBlock([
                    self.down,
                    ConvBlock(cs[1], cs[2]),
                    ConvBlock(cs[2], cs[2]),
                    SkipBlock([
                        self.down,
                        ConvBlock(cs[2], cs[3]),
                        ConvBlock(cs[3], cs[3]),
                        SkipBlock([
                            self.down,
                            ConvBlock(cs[3], cs[4]),
                            ConvBlock(cs[4], cs[4]),
                            SkipBlock([
                                self.down,
                                ConvBlock(cs[4], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[5]),
                                ConvBlock(cs[5], cs[4]),
                                self.up,
                            ]),
                            ConvBlock(cs[4] * 2, cs[4]),
                            ConvBlock(cs[4], cs[3]),
                            self.up,
                        ]),
                        ConvBlock(cs[3] * 2, cs[3]),
                        ConvBlock(cs[3], cs[2]),
                        self.up,
                    ]),
                    ConvBlock(cs[2] * 2, cs[2]),
                    ConvBlock(cs[2], cs[1]),
                    self.up,
                ]),
                ConvBlock(cs[1] * 2, cs[1]),
                ConvBlock(cs[1], cs[0]),
                self.up,
            ]),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


# ---- loading ----
def load_unet(cfg, hooks):
    name, steps = cfg["diffusion_model"], cfg["steps"]
    path = models.ensure(name, hooks.status)
    config = model_and_diffusion_defaults()
    config.update(models.config(name))
    config.update({
        "diffusion_steps": (1000 // steps) * steps if steps < 1000 else steps,
        "timestep_respacing": f"ddim{steps}",
        # With the secondary model the UNet never runs backward, so checkpointing does nothing useful.
        "use_checkpoint": not cfg["use_secondary_model"],
    })
    model, diffusion = create_model_and_diffusion(**config)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    # Disco Diffusion turns gradients on for the norm, qkv and proj weights. Nothing uses them, they cost
    # memory in every backward pass, and memory.lean_unet needs every weight frozen.
    model.requires_grad_(False).eval()
    model.convert_to_fp16()  # before moving, so the 32-bit weights never reach the GPU
    return model.to(device), diffusion


def load_secondary(hooks):
    model = SecondaryDiffusionImageNet2()
    model.load_state_dict(torch.load(models.ensure("secondary", hooks.status), map_location="cpu"))
    return model.eval().requires_grad_(False).to(device)


def load_clip_models(cfg):
    weights = paths.local_settings().get("vit_l14_weights", [])
    loaded = []
    for key, name in CLIP_NAMES.items():
        if cfg.get(key):
            model = memory.load_vit_l14(clip, weights) if key == "ViTL14" else clip.load(name, jit=False)[0]
            loaded.append(model.eval().requires_grad_(False).to(device))
    if not loaded:
        raise ValueError("Choose at least one CLIP model (prompt checker).")
    return loaded


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


IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_PROMPT_IMAGES = 200


def prompt_images(folder):
    """The images in an image prompt folder, at most MAX_PROMPT_IMAGES of them, in name order."""
    if not folder:
        return []
    if not os.path.isdir(folder):
        raise ValueError("The image prompt folder can't be found. Choose it again under Prompt.")
    found = sorted(os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(IMAGE_TYPES))
    if not found:
        raise ValueError("The image prompt folder has no PNG, JPEG or WebP images.")
    return found[:MAX_PROMPT_IMAGES]


@torch.no_grad()
def image_prompt_embed(model, files):
    """What a folder of images has in common, as one CLIP model sees it: the average of every image's
    embedding, each read whole and mirrored. Like a learned embedding, but nothing is trained."""
    side = model.visual.input_resolution
    total = None
    for path in files:
        image = ImageOps.fit(upright(path), (side, side), Image.LANCZOS)
        batch = TF.to_tensor(image).unsqueeze(0)
        batch = normalize(torch.cat([batch, batch.flip(3)])).to(device)
        embeds = F.normalize(model.encode_image(batch).float(), dim=-1).sum(0, keepdim=True)
        total = embeds if total is None else total + embeds
    return F.normalize(total, dim=-1)


def encode_prompts(clip_models, prompts, cfg, image_embeds=None):
    stats = []
    for index, model in enumerate(clip_models):
        embeds, weights = [], []
        if image_embeds:
            embeds.append(image_embeds[index])
            weights.append(float(cfg["image_prompt_weight"]))
        for prompt in prompts:
            text, weight = parse_prompt(prompt)
            embed = model.encode_text(clip.tokenize(text, truncate=True).to(device)).float()
            if cfg["fuzzy_prompt"]:
                for _ in range(25):
                    embeds.append((embed + torch.randn(embed.shape, device=device) * cfg["rand_mag"]).clamp(0, 1))
                    weights.append(weight)
            else:
                embeds.append(embed)
                weights.append(weight)
        weights = torch.tensor(weights, device=device)
        if weights.sum().abs() < 1e-3:
            raise ValueError("The prompt weights must not add up to 0.")
        stats.append({"model": model, "embeds": torch.cat(embeds), "weights": weights / weights.sum().abs()})
    return stats


def next_run_number(folder, batch_name):
    number = 0
    while os.path.exists(os.path.join(folder, f"{batch_name}({number})_settings.txt")):
        number += 1
    return number


class OnePassUNet:
    """Stops ddim running the UNet twice a step.

    The sampler asks for a prediction, then guidance asks for the same prediction with gradients.
    This does the second kind the first time and hands both callers the same result.
    """

    def __init__(self, model):
        self.model, self.leaf, self.output = model, None, None

    def parameters(self):
        return self.model.parameters()  # the sampler reads the device from these

    def __call__(self, x, t, **kwargs):
        if x is self.leaf:
            return self.output
        self.leaf = self.output = None
        if x.shape[-1] * x.shape[-2] >= 500_000:
            # Big renders fragment VRAM a little more every step. At 1280x768, 2.2GB sat reserved but
            # unusable by step 19, so each step hands the cached memory back and starts clean (about 5ms).
            torch.cuda.empty_cache()
        with torch.enable_grad():
            self.leaf = x.detach().requires_grad_()
            self.output = self.model(self.leaf, t, **{k: v for k, v in kwargs.items() if v is not None})
        return self.output


# ---- rendering ----
def set_seeds(seed):
    seed %= 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render(job_settings, prompts, hooks, offload=False, init_fit="crop", output_dir=paths.OUTPUT):
    cfg = {**DEFAULTS, **job_settings}
    style = models.cut_style(cfg["diffusion_model"])
    if style:
        cfg.update({k: v for k, v in models.CUT_SCHEDULES[style].items() if k not in job_settings})
    side_x, side_y = [(side // 64) * 64 for side in cfg.get("width_height", [768, 512])]
    secondary = cfg["use_secondary_model"]
    grid, overlap = tiles.grid_of(cfg["tile_scale"]), min(int(cfg["tile_overlap"]), min(side_x, side_y) // 2)
    final, spots = tiles.plan(side_x, side_y, grid, overlap)
    clash = rules.conflicts(cfg)
    if clash:
        raise ValueError(clash[0])
    trying = lab.wanted(cfg)
    animating = cfg["animation_mode"] != "None"
    mirrored = cfg["use_horizontal_symmetry"] or cfg["use_vertical_symmetry"]
    if trying and (grid > 1 or secondary or mirrored or cfg["diffusion_sampling_mode"] != "ddim"):
        raise ValueError("Experimental settings need the ddim sampler, with no tiles, symmetry or secondary model.")
    if grid > 1:
        cfg["diffusion_sampling_mode"] = "ddim"  # plms can't hold finished pixels still
    batch_name = cfg.get("batch_name", "Disco")
    folder = os.path.join(output_dir, batch_name)
    os.makedirs(folder, exist_ok=True)

    seed = random.randint(0, 2**32 - 1) if cfg["set_seed"] == "random_seed" else int(cfg["set_seed"])
    set_seeds(seed)
    sets = keyframes.prompt_sets(prompts) if animating else [(0, prompts)]
    handing = None
    if cfg["handoff_file"] and float(cfg["handoff_share"]) > 0:
        import handoff  # loaded before Disco, so its text encoders can use the GPU and leave
        handing = handoff.Handoff(cfg, sets, hooks.status, device, getattr(hooks, "stopped", lambda: False))
    torch.backends.cudnn.benchmark = False  # same speed here, but its cached workspaces cost VRAM

    overview = schedule(cfg["cut_overview"])
    innercut = schedule(cfg["cut_innercut"])
    ic_pow = schedule(cfg["cut_ic_pow"])
    icgray_p = schedule(cfg["cut_icgray_p"])

    clip_models = load_clip_models(cfg)
    model, diffusion = load_unet(cfg, hooks)
    secondary_model = load_secondary(hooks) if secondary else None
    pictures = prompt_images(cfg["image_prompts"]) if float(cfg["image_prompt_weight"]) else []
    if pictures:
        hooks.status(f"Reading {len(pictures)} prompt images")
    image_embeds = [image_prompt_embed(m, pictures) for m in clip_models] if pictures else None
    all_stats = [(at, encode_prompts(clip_models, lines, cfg, image_embeds)) for at, lines in sets]
    stats = all_stats[0][1]
    memory.free_text_encoders(clip_models)

    start_image = cfg.get("init_image") or None
    init_scale, skip_steps = cfg["init_scale"], cfg["skip_steps"]
    init, lpips_model = None, None
    if start_image:
        source = size_start_image(upright(start_image), *final, init_fit)
        if init_scale:
            import lpips  # loaded only when used: it takes VRAM and a few seconds
            lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device)

    drawing = cfg["live_ghost"] != "Off"
    if cfg["prompt_graft"] and cfg["partner_model"] != "Tiny SD":
        raise ValueError("Prompt graft needs Tiny SD as the partner model.")
    frame_ghost = animating and cfg["frame_ghost"]
    if cfg["depth_target"] and not (drawing or cfg["ghost_image"] or frame_ghost):
        raise ValueError("Depth target follows a ghost. Set Live ghost, or choose a Ghost image.")
    experiment = lab.Lab(cfg, schedule) if trying else None
    if uses_partner(cfg):
        text = ", ".join(parse_prompt(p)[0] for p in sets[0][1] if parse_prompt(p)[1] > 0)
        experiment.partner = partner.Partner(cfg["partner_model"], cfg["partner_file"], text, side_y, side_x,
                                             cfg["partner_follow"], device, hooks.status)
        partner.install(experiment.partner)
    if cfg["prompt_graft"]:
        experiment.graft = graft.Graft(schedule(cfg["graft_curve"]), device)
        experiment.graft.install(model)
        experiment.partner.watch(graft.sd_blocks(experiment.partner.unet))
    if cfg["ghost_image"] or cfg["style_image"] or drawing or frame_ghost:
        def sized(path):
            image = size_start_image(upright(path), side_x, side_y, "crop")
            return TF.to_tensor(image).unsqueeze(0).to(device)

        experiment.ghost = lab.Ghost(cfg, schedule, sized(cfg["ghost_image"]) if cfg["ghost_image"] else None,
                                     sized(cfg["style_image"]) if cfg["style_image"] else None, [s["model"] for s in stats], normalize,
                                     depth.Depth(device, hooks.status) if cfg["depth_target"] else None)

    progress = {"t": None}
    given = {"ghost": None, "sketch": None}  # in animation, a ghost made from the last frame, and the clip's one sketch
    place = None  # while tiling, puts the tile into a small copy of the whole picture
    pull = {"set": False, "to": None}  # after a handoff, Init scale still compares against the start before it
    one_pass = OnePassUNet(model) if cfg["diffusion_sampling_mode"] == "ddim" and not secondary else None

    def cond_fn(x, t, y=None):
        cur_t = progress["t"]
        with torch.enable_grad():
            x = one_pass.leaf if one_pass else x.detach().requires_grad_()
            n = x.shape[0]
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            if secondary:
                alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                sigma = torch.tensor(fac, device=device, dtype=torch.float32)
                pred = secondary_model(x, alpha_sigma_to_t(alpha, sigma)[None].repeat([n])).pred
            else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                pred = diffusion.p_mean_variance(one_pass or model, x, my_t, clip_denoised=False,
                                                 model_kwargs={"y": y})["pred_xstart"]
            x_in = pred * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            at = 1000 - (int(t.item()) + 1)
            guidance = cfg["clip_guidance_scale"] * experiment.guidance[at] if experiment else cfg["clip_guidance_scale"]
            for stat in stats:
                for _ in range(cfg["cutn_batches"]):
                    cuts = MakeCutoutsDango(stat["model"].visual.input_resolution, int(overview[at]), int(innercut[at]),
                                            ic_pow[at], icgray_p[at], cfg["skip_augs"], cfg["exact_repeat"],
                                            experiment.evidence if experiment else None)
                    whole = place(x_in).add(1).div(2) if place and int(overview[at]) else None
                    batch = normalize(cuts(x_in.add(1).div(2), whole))
                    total = int(overview[at]) + int(innercut[at])
                    chunk = memory.clip_chunk(stat["model"]) * n
                    if chunk >= batch.shape[0]:
                        image_embeds = stat["model"].encode_image(batch).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), stat["embeds"].unsqueeze(0))
                        dists = dists.view([total, n, -1])
                        losses = dists.mul(stat["weights"]).sum(2).mean(0)
                        x_in_grad += torch.autograd.grad(losses.sum() * guidance, x_in)[0] / cfg["cutn_batches"]
                    else:  # a few cuts at a time through a big CLIP, then one pass back through the cutting
                        leaf, cut_grad, parts = batch.detach().requires_grad_(), 0, []
                        for i in range(0, leaf.shape[0], chunk):
                            image_embeds = stat["model"].encode_image(leaf[i:i + chunk]).float()
                            part = spherical_dist_loss(image_embeds.unsqueeze(1), stat["embeds"].unsqueeze(0)).view([-1, n, len(stat["weights"])])
                            loss = part.mul(stat["weights"]).sum(2).sum(0).sum() / total
                            cut_grad = cut_grad + torch.autograd.grad(loss * guidance, leaf)[0]
                            parts.append(part.detach())
                        dists = torch.cat(parts)
                        x_in_grad += torch.autograd.grad(batch, x_in, cut_grad)[0] / cfg["cutn_batches"]
                    if cuts.evidence:
                        cuts.evidence.add(cuts.boxes, dists.mul(stat["weights"]).sum(2)[int(overview[at]):, 0])
            if experiment:
                x_in_grad = experiment.shape_push(x_in_grad, at)
                if experiment.ghost:
                    x_in_grad = experiment.ghost.push(x_in, x_in_grad, at)
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            # Disco Diffusion measures range on pred, which x_in is made from, so its gradient with respect to x_in
            # is always zero and Color range does nothing. Measured on x_in, it pulls colors back into range.
            loss = (tv_loss(x_in).sum() * cfg["tv_scale"] + range_loss(x_in).sum() * cfg["range_scale"]
                    + sat_losses.sum() * cfg["sat_scale"])
            if lpips_model is not None and init is not None and init_scale:
                loss = loss + lpips_distance(lpips_model, x_in, init).sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any():
                return torch.zeros_like(x)
            if offload:
                memory.park(clip_models)
            grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            if offload:
                memory.unpark(clip_models, device)
        if cfg["clamp_grad"]:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=cfg["clamp_max"]) / magnitude
        return grad

    def symmetry(x):
        if cfg["use_horizontal_symmetry"]:
            w = x.shape[-1]
            x = torch.concat((x[:, :, :, :w // 2], torch.flip(x[:, :, :, :w // 2], [-1])), -1)
        if cfg["use_vertical_symmetry"]:
            h = x.shape[-2]
            x = torch.concat((x[:, :, :h // 2, :], torch.flip(x[:, :, :h // 2, :], [-2])), -2)
        return x

    run = next_run_number(folder, batch_name)
    resume = cfg.get("resume_clip") if animating else None  # the last frame of a clip to carry on
    if resume:
        match = re.fullmatch(r"(.*)\((\d+)\)_0\.png", os.path.basename(resume))
        if not match:
            raise ValueError("Only an animation's own image can be continued.")
        folder, batch_name, run = os.path.dirname(resume), match.group(1), int(match.group(2))

    def write_notes(again=False):
        notes = os.path.join(folder, f"{batch_name}({run})_settings.txt")
        if again or not os.path.exists(notes):
            with open(notes, "w", encoding="utf-8") as f:
                json.dump({**cfg, "text_prompts": {str(at): lines for at, lines in sets}, "seed": seed, "init_fit": init_fit,
                           "offload": offload, "range_scale_works": True, "width_height": [side_x, side_y], "final_size": list(final)}, f, indent=2)

    def save(image, batch):
        path = os.path.join(folder, f"{batch_name}({run})_{batch}.png")
        write_notes()
        image.save(path)
        return path

    def to_image(tensor):
        return TF.to_pil_image(tensor[0].add(1).div(2).clamp(0, 1).cpu())

    def denoise(start, width, height, skip, batch, mask=None, show=to_image):
        """Runs one render and returns its finished image as a -1 to 1 tensor."""
        nonlocal init
        init = pull["to"] if pull["set"] else start  # Init scale compares against this
        painting_tile = mask is not None
        common = dict(clip_denoised=cfg["clip_denoised"], model_kwargs={}, cond_fn=cond_fn, progress=False,
                      skip_timesteps=skip, init_image=start, randomize_class=cfg["randomize_class"])
        progress["t"] = diffusion.num_timesteps - skip - 1
        total_steps = progress["t"]
        shape = (1, 3, height, width)
        if cfg["diffusion_sampling_mode"] == "ddim":
            samples = diffusion.ddim_sample_loop_progressive(
                one_pass or model, shape, eta=cfg["eta"], transformation_fn=None if painting_tile else symmetry,
                transformation_percent=cfg["transformation_percent"], inpainting_mode=painting_tile, mask_inpaint=mask, **common)
        else:
            samples = diffusion.plms_sample_loop_progressive(model, shape, order=2, **common)
        for j, sample in enumerate(samples):
            progress["t"] -= 1
            last = progress["t"] == -1
            hooks.step(j, total_steps, batch, n_batches, seed)
            if j % cfg["display_rate"] == 0 or last:
                hooks.preview(show(sample["pred_xstart"]))
        release()
        return sample["pred_xstart"].detach()

    def denoise_lab(start, width, height, skip, batch):
        """The normal ddim loop, with room for the experiments that work between steps."""
        nonlocal init
        init = pull["to"] if pull["set"] else start
        experiment.reset(height, width, device)
        helper = experiment.partner
        ghost = experiment.ghost
        needed = cfg["two_painters"] or cfg["live_ghost"] == "Live" or cfg["borrow_attention"] or cfg["prompt_graft"]
        grafting = experiment.graft
        sketching = helper and cfg["live_ghost"] == "Sketch first" and given["ghost"] is None
        if sketching and given["sketch"] is not None:  # a clip keeps its first sketch, or the layout would jump every frame
            ghost.ghost, sketching = given["sketch"], False
        if helper and (needed or sketching):
            helper.to(device)
            helper.restart(seed + batch)
            if sketching:
                hooks.status("Painting the sketch")
                drawn = helper.paint(cfg["sketch_steps"])
                ghost.ghost = drawn.add(1).div(2).float()
                if animating:
                    given["sketch"] = ghost.ghost
                if cfg["partner_keep_sketch"]:
                    save_sketch(drawn)
            if not needed:  # only the sketch was wanted, so the render gets the VRAM back
                helper.to("cpu")
                torch.cuda.empty_cache()
        if given["ghost"] is not None:
            ghost.ghost = given["ghost"]
        first = diffusion.num_timesteps - skip - 1
        x = torch.randn(1, 3, height, width, device=device)
        if skip or start is not None:
            base = torch.zeros_like(x) if start is None else start
            x = diffusion.q_sample(base, torch.tensor([first], device=device), x)
        image = torch.zeros_like(x)
        for j, i in enumerate(range(first, -1, -1)):
            progress["t"] = i
            hooks.step(j, first, batch, n_batches, seed)
            t = torch.tensor([i], device=device)
            at = min(int(1000 * (1 - (i + 1) / diffusion.num_timesteps)), 999)
            band = int(experiment.band[at])
            if helper and needed:
                helper.strength = experiment.borrow[at]
                live = cfg["live_ghost"] == "Live" and ghost.curve[at]
                if grafting:
                    grafting.strength = cfg["prompt_graft"] * grafting.curve[at]
                if band or (helper.strength or live or grafting and grafting.strength) and j % cfg["partner_every"] == 0:
                    helper.refresh(image, float(diffusion.alphas_cumprod[i]))
                    if live:
                        ghost.ghost = helper.sketch().add(1).div(2).float()
                    if grafting:
                        grafting.update(helper.difference, helper.nearest(float(diffusion.alphas_cumprod[i])))
            alpha, alpha_prev = diffusion.alphas_cumprod[i], diffusion.alphas_cumprod_prev[i]
            repeats = experiment.repeats(at)
            for repeat in range(repeats + 1):
                with torch.no_grad():
                    out = diffusion.ddim_sample(one_pass, x, t, clip_denoised=cfg["clip_denoised"], cond_fn=cond_fn,
                                                model_kwargs={}, eta=cfg["eta"])
                x = out["sample"]
                if repeat < repeats:  # back up one step, so the model can absorb the push
                    ratio = alpha / alpha_prev
                    x = math.sqrt(ratio) * x + math.sqrt(1 - ratio) * torch.randn_like(x)
            image = out["pred_xstart"].detach()
            if ghost:
                x = ghost.pull(x, image, math.sqrt(alpha_prev), at)
            if band:
                moved = experiment.share(helper.sketch(), image, band)
                image = image + moved
                x = x + math.sqrt(alpha_prev) * moved
                if experiment.band_path == "Blend":  # the noise estimate changes with the mix, so the step stays small
                    sigma = cfg["eta"] * math.sqrt((1 - alpha_prev) / (1 - alpha)) * math.sqrt(1 - alpha / alpha_prev)
                    x = x - math.sqrt(max(1 - alpha_prev - sigma ** 2, 0)) * math.sqrt(alpha / (1 - alpha)) * moved
            if j % cfg["display_rate"] == 0 or not i:
                hooks.preview(to_image(image))
        if helper and needed and helper.z0 is not None and cfg["partner_keep_sketch"]:
            save_sketch(helper.sketch())
        release()
        return image

    def release():
        """The last step's UNet output keeps its whole graph alive until the next step, so let it go."""
        if one_pass:
            one_pass.leaf = one_pass.output = None

    def hand_over(start, skip, n, size, noise_seed, keep_as=None):
        """With a handoff model, it paints its share of the path from `start` (a -1 to 1 tensor, or None
        for noise) and Disco carries on from there. Returns the new start and skip. `size` is the size
        painted when there is no start."""
        pull["set"], pull["to"] = False, None
        if handing is None:
            return start, skip
        release()
        begins = float(diffusion.alphas_cumprod[diffusion.num_timesteps - skip - 1])
        picture = None if start is None else to_image(start)
        resting = [model, secondary_model, lpips_model, *(m.visual for m in clip_models),
                   experiment.partner if experiment else None]
        with handoff.Parked(handing, resting, device, *size):
            made = handing.paint(picture, begins, n, *size, noise_seed)
        if made is None:
            return start, skip
        image, alpha = made
        if keep_as:
            image.save(keep_as)
        pull["set"], pull["to"] = True, start
        skip = handoff.disco_skip(diffusion.alphas_cumprod, alpha, skip)
        hooks.status(f"Disco carries on from step {skip + 1} of {cfg['steps']}")
        return TF.to_tensor(image).unsqueeze(0).mul(2).sub(1).to(device), skip

    def save_sketch(sketch):
        to_image(sketch).save(os.path.join(folder, f"{batch_name}({run})_partner.png"))

    def paint_tiles(start, skip, batch):
        """Takes every tile through every step together, on one picture held in RAM.

        Each step, every tile predicts its part and the overlaps are averaged before the picture moves
        on, so neighbors settle on the same image instead of being joined at the end. Each tile is an
        ordinary render as far as the GPU is concerned.
        """
        nonlocal place, init
        size = (1, 3, final[1], final[0])
        noise = torch.randn(size)
        first = diffusion.num_timesteps - skip - 1
        canvas = noise if skip == 0 and start is None else None
        if canvas is None:
            canvas = diffusion.q_sample(torch.zeros(size) if start is None else start, torch.tensor([first]), noise)
        latest = start
        for j, i in enumerate(range(first, -1, -1)):
            progress["t"] = i
            t = torch.tensor([i], device=device)
            view = None if latest is None else tiles.small_copy(latest)
            pred_sum, eps_sum = torch.zeros(size), torch.zeros(size)
            across = tiles.fades(final[0], side_x - overlap, grid, overlap)
            down = tiles.fades(final[1], side_y - overlap, grid, overlap)
            for number, (x, y) in enumerate(spots):
                hooks.step(j * len(spots) + number, (first + 1) * len(spots), batch, n_batches, seed)
                box = (..., slice(y, y + side_y), slice(x, x + side_x))
                weight = down[number // grid][y:y + side_y, None] * across[number % grid][x:x + side_x]
                tile = canvas[box].to(device)
                place = view and cfg["tile_whole_view"] and tiles.placer(*view, x, y, side_x, side_y, device)
                held = pull["to"] if pull["set"] else start
                init = None if held is None else held[box].to(device)  # Init scale compares against this
                with torch.no_grad():
                    out = diffusion.p_mean_variance(one_pass or model, tile, t, clip_denoised=cfg["clip_denoised"], model_kwargs={})
                    pred = diffusion.condition_score(cond_fn, out, tile, t, model_kwargs={})["pred_xstart"]
                    eps = diffusion._predict_eps_from_xstart(tile, t, pred)
                    if not i:
                        pred = out["pred_xstart"]  # as in a normal render, the last image is the model's own
                pred_sum[box] += pred.cpu() * weight
                eps_sum[box] += eps.cpu() * weight
            place = None
            # One DDIM step on the whole picture, so its noise doesn't get averaged away in the overlaps.
            alpha, alpha_prev = diffusion.alphas_cumprod[i], diffusion.alphas_cumprod_prev[i]
            sigma = cfg["eta"] * math.sqrt((1 - alpha_prev) / (1 - alpha)) * math.sqrt(1 - alpha / alpha_prev)
            canvas = pred_sum * math.sqrt(alpha_prev) + math.sqrt(1 - alpha_prev - sigma ** 2) * eps_sum
            if i:
                canvas += sigma * torch.randn(size)
            latest = pred_sum.clamp(-1, 1)
            if j % max(cfg["display_rate"] // len(spots), 1) == 0 or not i:
                preview = to_image(latest)
                preview.thumbnail((1024, 1024))  # saving a full-size preview takes longer than it's worth
                hooks.preview(preview)
        release()
        return latest

    def film():
        """Renders the frames of an animation, each from the last one moved by the camera or from a video frame."""
        nonlocal stats, init_scale, lpips_model, n_batches
        mode = cfg["animation_mode"]
        write_notes(again=bool(resume))  # claims the run number, so a stopped animation keeps its frames
        frames_dir = os.path.join(folder, f"{batch_name}({run})_frames")
        os.makedirs(frames_dir, exist_ok=True)
        video = None
        if mode == "Video Input":
            hooks.status("Reading the video")
            video = animate.VideoInput(cfg["video_init_path"], max(int(cfg["extract_nth_frame"]), 1), *final,
                                       init_fit, size_start_image, int(cfg["max_frames"]))
        total = min(int(cfg["max_frames"]), len(video.frames)) if video else int(cfg["max_frames"])
        n_batches = total
        depth_model = experiment.ghost.depth if experiment and experiment.ghost else None  # one copy serves both
        camera = animate.Camera(cfg, total, device, hooks.status, depth_model) if mode in ("2D", "3D") else None
        skips = [min(round(cfg["steps"] * share / 100), cfg["steps"] - 1) for share in keyframes.skips(cfg, total)]
        if cfg["frames_scale"] and lpips_model is None:
            import lpips
            lpips_model = lpips.LPIPS(net="vgg", verbose=False).to(device)
        whole = denoise_lab if experiment else denoise

        def paint(start, skip, n, hand=True):
            """One frame. With tiles, each frame is tiled like a still, and a start uses Tile denoise."""
            if hand and handing:
                start, skip = hand_over(start, skip, n, (side_x, side_y) if grid == 1 else final, frame_seed)
                set_seeds(frame_seed)  # each frame starts alike, whatever the handoff drew
            if grid == 1:
                return whole(start, side_x, side_y, skip, n)
            return paint_tiles(None if start is None else start.float().cpu(), skip, n).to(device)

        # A frame that starts from an image or video rather than the last frame uses Denoise, or Tile denoise with tiles.
        start_skip = tile_skip if grid > 1 else skip_steps
        turbo = max(int(cfg["turbo_steps"]), 1) if cfg["turbo_mode"] and camera else 1
        refresh = {at + int(cfg["prompt_fade"]) for at, _ in all_stats[1:]}  # a new prompt gets its own colors
        first, previous, saved = None, None, {}
        rendered_first = False  # the first frame's Render first pass already had its handoff
        between, painted_at = [], 0  # with Turbo, the frames moved on from the last repaint
        begin = 0
        if resume:  # pick up after the last frame on disk, as if the run had never stopped
            while os.path.isfile(os.path.join(frames_dir, f"{begin:05d}.png")):
                saved[begin] = os.path.join(frames_dir, f"{begin:05d}.png")
                begin += 1
            if not begin:
                raise ValueError("That clip has no frames to continue from.")
            if begin >= total:
                raise ValueError(f"That clip already has {begin} frames. Raise Frames to make it longer.")

            def frame_at(n):
                return TF.to_tensor(Image.open(saved[n]).convert("RGB")).unsqueeze(0).mul(2).sub(1).to(device)

            previous = frame_at(begin - 1)
            if tuple(previous.shape[-2:]) != (final[1], final[0]):
                raise ValueError(f"That clip's frames are {previous.shape[-1]}x{previous.shape[-2]}. Set the same size and Tiles "
                                 "to carry it on.")
            first = frame_at(max([0] + [n for n in refresh if n < begin]))
            painted_at = begin - 1
            hooks.status(f"Carrying on from frame {begin + 1}")

        def keep(n, image):
            saved[n] = os.path.join(frames_dir, f"{n:05d}.png")
            to_image(image).save(saved[n])

        try:
            for n in range(begin, total):
                stats = animate.stats_at(all_stats, n, cfg["prompt_fade"])
                # Only a video can share one noise: a moving camera drags fixed noise across the scene into streaks.
                frame_seed = seed if video and cfg["video_init_seed_continuity"] else seed + n
                set_seeds(frame_seed)
                if video:
                    shown = TF.to_tensor(video.frames[n]).unsqueeze(0).mul(2).sub(1).to(device)
                    start = shown
                    if previous is None and cfg["video_blend"]:
                        # The first frame builds on a first pass of itself, as later frames build on the last
                        # one. Without this it stays close to the video while the rest of the clip doesn't.
                        hooks.status("Painting a first pass of the first frame")
                        previous = paint(shown, start_skip, n)  # a full repaint, as a still would be
                    if previous is not None and cfg["video_blend"]:
                        last = previous
                        if cfg["video_flow"] and n:  # the last frame's paint travels with what moved in the video
                            last = animate.follow_flow(previous, animate.video_flow(video.frames[n - 1], video.frames[n]), cfg)
                        start = shown + cfg["video_blend"] * (last - shown)
                    skip = skips[n]
                elif n == 0:
                    shown = start = TF.to_tensor(source).unsqueeze(0).mul(2).sub(1).to(device) if start_image else None
                    skip = start_skip if start_image else skip_steps
                    if grid > 1 and not start_image and cfg["tile_start"] == "Render first":
                        hooks.status("Painting the first frame at tile size")
                        rough, rough_skip = hand_over(None, skip_steps, n, (side_x, side_y), frame_seed)
                        set_seeds(frame_seed)
                        rough = to_image(denoise(rough, side_x, side_y, rough_skip, n)).resize(final, Image.LANCZOS)
                        shown = start = TF.to_tensor(rough).unsqueeze(0).mul(2).sub(1).to(device)
                        skip, rendered_first = tile_skip, True
                else:
                    moved = animate.sharpen(camera.move(previous, n), cfg["frame_sharpen"])
                    shown = start = animate.hold_colors(moved, first, cfg["color_hold"])
                    skip = skips[n]
                init_scale = 0 if start is None else cfg["frames_scale"] if (n or video) else cfg["init_scale"]
                given["ghost"] = shown.add(1).div(2).float() if frame_ghost and shown is not None else None
                if turbo > 1 and n % turbo and n < total - 1:
                    between.append((n, start))  # between repaints the camera alone moves the picture
                    previous = start
                    hooks.preview(to_image(start))
                    continue
                gc.collect()
                torch.cuda.empty_cache()
                set_seeds(frame_seed)  # passes before this one drew noise too, and every frame must start alike
                image = paint(start, skip, n, hand=not rendered_first)
                rendered_first = False
                back = image
                for m, ahead in reversed(between):  # fade each one toward the new repaint, moved back to it
                    back = camera.move(back, m + 1, back=True)
                    keep(m, ahead + (m - painted_at) / (n - painted_at) * (back - ahead))
                between, painted_at = [], n
                previous = image
                first = image if first is None or n in refresh else first
                keep(n, image)
        finally:  # a stopped animation still gets a video of the frames it has
            for m, ahead in between:
                keep(m, ahead)
            saved = [saved[n] for n in sorted(saved)]
            if saved:
                fps = max(float(cfg["fps"]), 1)
                extra = int(cfg["in_between"])
                if extra > 0 and len(saved) > 1:  # the clip keeps its length, at a higher frame rate
                    saved = animate.add_between(saved, extra, os.path.join(frames_dir, "between"), device, hooks.status)
                    fps *= extra + 1
                hooks.status("Saving the video")
                base = os.path.join(folder, f"{batch_name}({run})")
                if cfg["save_mp4"] and len(saved) > 1:
                    animate.save_video(saved, fps, base + ".mp4")
                if cfg["save_webp"] and len(saved) > 1:
                    animate.save_animation(saved, fps, base + ".webp")
                if cfg["save_gif"] and len(saved) > 1:
                    animate.save_animation(saved, fps, base + ".gif", longest=512)
                hooks.image(save(to_image(previous), 0))

    n_batches = cfg["n_batches"]
    hooks.status("Starting the first step")
    tile_skip = min(round(cfg["steps"] * (1 - cfg["tile_denoise"])), cfg["steps"] - 1)
    if animating:
        film()
        return
    for batch in range(n_batches):
        gc.collect()
        torch.cuda.empty_cache()
        set_seeds(seed + batch)  # each image can be made again on its own
        start = TF.to_tensor(source).unsqueeze(0).mul(2).sub(1).to(device) if start_image else None
        kept = cfg["handoff_keep"] and os.path.join(folder, f"{batch_name}({run})_{batch}_handoff.png")
        if grid == 1:
            paint = denoise_lab if experiment else denoise
            start, skip = hand_over(start, skip_steps, batch, (side_x, side_y), seed + batch, kept)
            set_seeds(seed + batch)
            image = to_image(paint(start, side_x, side_y, skip, batch))
        else:
            if not start_image and cfg["tile_start"] == "Render first":
                start, skip = hand_over(None, skip_steps, batch, (side_x, side_y), seed + batch, kept)
                set_seeds(seed + batch)
                first = to_image(denoise(start, side_x, side_y, skip, batch))
                start, skip = TF.to_tensor(first.resize(final, Image.LANCZOS)).unsqueeze(0).mul(2).sub(1), tile_skip
            else:
                start, skip = hand_over(start, tile_skip if start is not None else skip_steps, batch, final, seed + batch, kept)
                set_seeds(seed + batch)
            image = to_image(paint_tiles(None if start is None else start.cpu(), skip, batch))
        hooks.image(save(image, batch))
