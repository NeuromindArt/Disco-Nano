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
import traceback
from dataclasses import dataclass
from functools import partial

import clip
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from guided_diffusion.script_util import create_gaussian_diffusion, create_model_and_diffusion, model_and_diffusion_defaults
from PIL import Image, ImageFilter, ImageOps
from resize_right import resize
from torch import nn
from torch.nn import functional as F

import animate
import control
import critic
import erase
import depth
import graft
import imaging
import keyframes
import lab
import memory
import models
import partner
import paths
import rules
import tiles
from keyframes import POINTS, schedule, schedule_at
from settings import DEFAULTS, handoff_parts, small_first_on, small_frame, uses_partner

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
# ---- models kept between renders ----
# The render worker stays open between renders, and keeps the models it loaded in RAM for the next one. Each
# render brings what it needs to the GPU, and put_away sends it all back to RAM when the render ends, so a
# waiting worker holds no VRAM beyond its CUDA context. Closing the app ends the worker and frees it all. Models the
# last render didn't use are let go, so switching checkpoints or settings doesn't pile up copies in RAM. A model only
# Fix's Inpaint used is let go too: it's needed now and then, and holds many GB that the next render may want.
_kept = {}
_used = set()  # keys asked for during this render, to keep for the next one
_after_render = []  # undoes what a render changed in kept models or shared code, such as hooks


def kept(key, build, brief=False):
    """The model kept under key, built the first time it is asked for. A `brief` one is let go when the render ends,
    unless the render also asked for it to keep."""
    if not brief:
        _used.add(key)
    if key not in _kept:
        _kept[key] = build()
    return _kept[key]


def put_away():
    while _after_render:
        try:
            _after_render.pop()()
        except Exception:  # every undo still runs, and what went wrong goes to the job's log
            traceback.print_exc()
    for key in [k for k in _kept if k not in _used]:
        del _kept[key]
    _used.clear()
    gc.collect()  # what was let go leaves RAM before the rest comes back to it from the GPU
    for item in _kept.values():
        for module in (item.parts() if hasattr(item, "parts") else [item]):
            module.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()


def load_unet(cfg, hooks):
    name, steps = cfg["diffusion_model"], cfg["steps"]
    config = model_and_diffusion_defaults()
    config.update(models.config(name))
    config.update({
        "diffusion_steps": (1000 // steps) * steps if steps < 1000 else steps,
        "timestep_respacing": f"ddim{steps}",
        # With the secondary model the UNet never runs backward, so checkpointing does nothing useful.
        "use_checkpoint": not cfg["use_secondary_model"],
    })

    def build():
        path = models.ensure(name, hooks.status)
        model, _ = create_model_and_diffusion(**config)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        # Disco Diffusion turns gradients on for the norm, qkv and proj weights. Nothing uses them, they cost
        # memory in every backward pass, and memory.lean_unet needs every weight frozen.
        model.requires_grad_(False).eval()
        model.convert_to_fp16()  # before moving, so the 32-bit weights never reach the GPU
        return model

    # The weights don't depend on the steps, so a kept model serves any number of them with a new schedule.
    model = kept(("unet", name, config["use_checkpoint"]), build)
    diffusion = create_gaussian_diffusion(
        steps=config["diffusion_steps"], learn_sigma=config["learn_sigma"], noise_schedule=config["noise_schedule"],
        use_kl=config["use_kl"], predict_xstart=config["predict_xstart"], rescale_timesteps=config["rescale_timesteps"],
        rescale_learned_sigmas=config["rescale_learned_sigmas"], timestep_respacing=config["timestep_respacing"])
    return model.to(device), diffusion


def load_secondary(hooks):
    def build():
        model = SecondaryDiffusionImageNet2()
        model.load_state_dict(torch.load(models.ensure("secondary", hooks.status), map_location="cpu"))
        return model.eval().requires_grad_(False)

    return kept(("secondary",), build).to(device)


def load_clip_models(cfg, status=print):
    weights = paths.local_settings().get("vit_l14_weights", [])
    loaded = []
    for key, name in CLIP_NAMES.items():
        if cfg.get(key):
            def build(key=key, name=name):
                model = memory.load_vit_l14(clip, weights) if key == "ViTL14" else clip.load(name, jit=False)[0]
                return model.eval().requires_grad_(False)

            loaded.append(kept(("clip", key), build).to(device))
    robust = cfg.get("robust_critic", "Off")
    if robust != "Off":
        loaded.append(kept(("clip", robust), lambda: critic.load_robust(robust, "cpu", status)).to(device))
    if not loaded:
        raise ValueError("Choose at least one CLIP model (prompt checker).")
    return loaded


def load_lpips():
    import lpips  # loaded only when used: it takes VRAM and a few seconds
    return kept(("lpips",), lambda: lpips.LPIPS(net="vgg", verbose=False).eval().requires_grad_(False)).to(device)


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


def encode_prompts(clip_models, prompts, cfg, image_embeds=None, roles=None):
    """roles, when prompts are routed by cut scale, says whether each line is subject or style."""
    stats = []
    for index, model in enumerate(clip_models):
        embeds, weights, kinds = [], [], []
        if image_embeds:
            embeds.append(image_embeds[index])
            weights.append(float(cfg["image_prompt_weight"]))
            kinds.append(critic.BOTH)
        for number, prompt in enumerate(prompts):
            text, weight = parse_prompt(prompt)
            embed = model.encode_text(clip.tokenize(text, truncate=True).to(device)).float()
            kind = roles[number] if roles else critic.BOTH
            if cfg["fuzzy_prompt"]:
                for _ in range(25):
                    embeds.append((embed + torch.randn(embed.shape, device=device) * cfg["rand_mag"]).clamp(0, 1))
                    weights.append(weight)
                    kinds.append(kind)
            else:
                embeds.append(embed)
                weights.append(weight)
                kinds.append(kind)
        weights = torch.tensor(weights, device=device)
        if weights.sum().abs() < 1e-3:
            raise ValueError("The prompt weights must not add up to 0.")
        stats.append({"model": model, "embeds": torch.cat(embeds), "weights": weights / weights.sum().abs(),
                      **({"roles": torch.tensor(kinds)} if roles else {})})
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
    """One render or clip. Whatever happens, the models it used go back to RAM for the next render."""
    try:
        _render(job_settings, prompts, hooks, offload, init_fit, output_dir)
    finally:
        put_away()


def _render(job_settings, prompts, hooks, offload, init_fit, output_dir):
    cfg = {**DEFAULTS, **job_settings}
    _lpips_pixels.clear()
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
    # Small first: the first small_steps of the path are painted at `small`, in the frame's shape.
    small = small_frame(side_x, side_y, int(cfg["small_side"])) if grid == 1 and small_first_on(cfg) else None
    small_steps = min(round(cfg["steps"] * float(cfg["small_first"]) / 100), cfg["steps"] - 1) if small else 0
    trying = lab.wanted(cfg) or critic.wanted(cfg)
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
    hand_start, hand_finish = handoff_parts(cfg)
    if hand_start or hand_finish:
        import handoff  # loaded before Disco, so its text encoders can use the GPU and leave
        path = cfg["handoff_file"]
        handing = kept(("handoff", path, os.path.getmtime(path)), lambda: handoff.load(path, hooks.status, device))
        handing.prepare(cfg, sets, hooks.status, getattr(hooks, "stopped", lambda: False), hooks.preview)
    torch.backends.cudnn.benchmark = False  # same speed here, but its cached workspaces cost VRAM

    overview = schedule(cfg["cut_overview"])
    innercut = schedule(cfg["cut_innercut"])
    ic_pow = schedule(cfg["cut_ic_pow"])
    icgray_p = schedule(cfg["cut_icgray_p"])
    # The views schedules lay out the scene in their first 40% and paint detail after. Shapes share spreads that
    # first part over a share of the steps, and Fit views spreads both over only the steps a picture paints, so one
    # that starts partway, from a start image or a video frame, still lays out shapes before its detail.
    views = {"from": 0}  # where on the 1000-point schedule the picture being painted began

    def fit_views(skip, frame=False):
        """`frame` is a clip's frame, which is fitted only with Start images and clip frames."""
        mode = cfg["views_fit"]
        if isinstance(mode, bool):  # settings saved while it was a switch
            mode = "Start images and clip frames" if mode else "Off"
        fit = mode == "Start images and clip frames" or (mode == "Start images" and not frame)
        views["from"] = min(round(POINTS * skip / diffusion.num_timesteps), POINTS - 1) if fit else 0

    def view_at(at):
        begin, shapes, pivot = views["from"], min(max(float(cfg["views_shapes"]) / 100, 0), 1), 400
        done = min(max((at - begin) / (1000 - begin), 0), 1)
        if done < shapes:
            point = done / shapes * pivot
        else:
            point = pivot + (done - shapes) / max(1 - shapes, 1e-6) * (1000 - pivot)
        return min(int(point), POINTS - 1)

    clip_models = load_clip_models(cfg, hooks.status)
    model, diffusion = load_unet(cfg, hooks)
    secondary_model = load_secondary(hooks) if secondary else None
    pictures = prompt_images(cfg["image_prompts"]) if float(cfg["image_prompt_weight"]) else []
    if pictures:
        hooks.status(f"Reading {len(pictures)} prompt images")
    image_embeds = [image_prompt_embed(m, pictures) for m in clip_models] if pictures else None
    judge = critic.Critic(cfg, schedule, seed) if critic.wanted(cfg) else None
    if judge and judge.route != "Off":
        routed = [(at, *critic.split_lines(lines, cfg["route_split"], parse_prompt)) for at, lines in sets]
        all_stats = [(at, encode_prompts(clip_models, lines, cfg, image_embeds, roles)) for at, lines, roles in routed]
    else:
        all_stats = [(at, encode_prompts(clip_models, lines, cfg, image_embeds)) for at, lines in sets]
    stats = all_stats[0][1]
    if judge:
        judge.prepare_subjects(stats, critic.subject_text(cfg, sets[0][1], parse_prompt), clip.tokenize, device)
    regions = {"rev": None, "boxes": [], "strength": 0.0, "embeds": {}, "dirty": False}

    def take_regions():
        """Regional prompts: boxes with prompts of their own. Encodes them at the start, and again when a pause
        changed them. Returns True when it changed them."""
        rev = getattr(hooks, "regions_rev", 0)
        if rev == regions["rev"]:
            return False
        given = hooks.regions if hasattr(hooks, "regions") else cfg.get("regions")
        boxes = [b for b in (given or {}).get("boxes", []) if str(b.get("text", "")).strip()]
        moving = regions["rev"] is not None and boxes  # the text encoders have left the GPU by now
        regions.update(rev=rev, boxes=boxes, strength=float((given or {}).get("strength", 0.75)), embeds={})
        if regions["rev"] or given:
            cfg["regions"] = given
            regions["dirty"] = True
        if moving:
            memory.text_encoders_back(clip_models, device)
        with torch.no_grad():
            for model in clip_models:
                tokens = clip.tokenize([parse_prompt(b["text"])[0] for b in boxes], truncate=True).to(device) if boxes else None
                if tokens is not None:
                    regions["embeds"][model] = model.encode_text(tokens).float()
        if moving:
            memory.free_text_encoders(clip_models)
        return True

    def region_rows(main, boxes, overview, total, height, width):
        """[cuts, prompts + regions] weights. A close-up is judged by each region's prompt in proportion to how much
        of it the region covers, times Region strength, and by the main prompts for the rest. A whole-image view
        is judged the same way, by how much of the frame each region covers.

        With Tiles, the regions lie over the whole tiled picture, and a tile's close-ups are measured where the tile
        is in it. A tile's whole-image view is the whole picture when Whole picture view shows the tile in it, and
        otherwise only the tile."""
        left, top = tile_at or (0, 0)
        full_w, full_h = final if tile_at else (width, height)
        rects = torch.tensor([[b["x"] * full_w, b["y"] * full_h, (b["x"] + b["w"]) * full_w, (b["y"] + b["h"]) * full_h]
                              for b in regions["boxes"]], device=device)
        seen = [0, 0, full_w, full_h] if tile_at is None or place else [left, top, left + width, top + height]
        spots = torch.tensor([seen] * overview + [[left + x, top + y, left + x + size, top + y + size] for x, y, size in boxes],
                             device=device, dtype=torch.float32)
        across = (torch.minimum(spots[:, None, 2], rects[None, :, 2]) - torch.maximum(spots[:, None, 0], rects[None, :, 0])).clamp(min=0)
        down = (torch.minimum(spots[:, None, 3], rects[None, :, 3]) - torch.maximum(spots[:, None, 1], rects[None, :, 1])).clamp(min=0)
        cover = across * down / ((spots[:, 2] - spots[:, 0]) * (spots[:, 3] - spots[:, 1]))[:, None]
        cover = cover / cover.sum(1, keepdim=True).clamp(min=1)  # overlapping regions share the cut
        share = regions["strength"] * cover
        main = main.expand(total, -1) if main.dim() == 1 else main
        return torch.cat([main * (1 - share.sum(1, keepdim=True)), share], 1)

    take_regions()
    memory.free_text_encoders(clip_models)

    start_image = cfg.get("init_image") or None
    init_scale, skip_steps = cfg["init_scale"], cfg["skip_steps"]
    init, lpips_model = None, None
    if start_image:
        source = size_start_image(upright(start_image), *final, init_fit)
        if init_scale:
            lpips_model = load_lpips()

    drawing = cfg["live_ghost"] != "Off"
    if cfg["prompt_graft"] and cfg["partner_model"] != "Tiny SD":
        raise ValueError("Prompt graft needs Tiny SD as the partner model.")
    frame_ghost = animating and cfg["frame_ghost"]
    if cfg["depth_target"] and not (drawing or cfg["ghost_image"] or frame_ghost):
        raise ValueError("Depth target follows a ghost. Set Live ghost, choose a Ghost image, or turn on Last frame as ghost in a clip.")
    experiment = lab.Lab(cfg, schedule) if trying else None

    def depth_model():
        """Depth Anything: one copy for Depth target, Depth control, Face shape and the 3D camera, on the GPU for the
        render and kept in RAM between renders, as Disco's other models are."""
        return kept(("depth",), lambda: depth.Depth(device, hooks.status)).to(device)

    if uses_partner(cfg):
        text = ", ".join(parse_prompt(p)[0] for p in sets[0][1] if parse_prompt(p)[1] > 0)
        source, file = cfg["partner_model"], cfg["partner_file"] if cfg["partner_model"] != "Tiny SD" else ""
        weights = kept(("partner", source, file, os.path.getmtime(file) if os.path.isfile(file) else 0),
                       lambda: partner.Weights(source, file, hooks.status))
        experiment.partner = partner.Partner(weights, text, side_y, side_x, cfg["partner_follow"], device)
        _after_render.append(experiment.partner.release)
        scales = critic.GRIDS.get(cfg["borrow_grids"], critic.GRIDS["All"])
        experiment.partner.lengths = {(side_y // s) * (side_x // s) for s in scales}
        partner.install(experiment.partner)
        _after_render.append(partner.uninstall)
    if cfg["prompt_graft"]:
        experiment.graft = graft.Graft(schedule(cfg["graft_curve"]), device)
        _after_render.append(experiment.graft.install(model))
        experiment.partner.watch(graft.sd_blocks(experiment.partner.unet))
    if cfg["ghost_image"] or cfg["style_image"] or drawing or frame_ghost:
        def sized(path):
            image = size_start_image(upright(path), side_x, side_y, "crop")
            return TF.to_tensor(image).unsqueeze(0).to(device)

        experiment.ghost = lab.Ghost(cfg, schedule, sized(cfg["ghost_image"]) if cfg["ghost_image"] else None,
                                     sized(cfg["style_image"]) if cfg["style_image"] else None, [s["model"] for s in stats], normalize,
                                     depth_model() if cfg["depth_target"] else None)

    steer = None  # the controls: edges, depth and pose held to a control picture
    fixed_control = None  # the Control image, when one is set; otherwise the start image or each frame's video frame
    if control.wanted(cfg):
        steer = control.Control(cfg, schedule, device, hooks.status, depth_model, kept)
        if cfg.get("control_image") and not animating:  # a clip follows each frame, never one fixed picture
            fixed_control = TF.to_tensor(size_start_image(upright(cfg["control_image"]), side_x, side_y, init_fit)).unsqueeze(0)

    progress = {"t": None}
    # in animation, a ghost made from the last frame, the clip's one sketch, and a frame's own start noise
    given = {"ghost": None, "sketch": None, "noise": None}

    def start_noise(shape, where):
        """The frame's own start noise, when it has one of this shape, or None for a fresh draw."""
        noise = given["noise"]
        if noise is None or tuple(noise.shape) != tuple(shape):
            return None
        return noise.to(where)
    place = None  # while tiling, puts the tile into a small copy of the whole picture
    tile_at = None  # while tiling, where the tile is in the whole picture, so regions measure its close-ups there
    pull = {"set": False, "to": None}  # after a handoff, Init scale still compares against the start before it
    ahead = {"steps": 0}  # steps Small first already painted, so the full-size pass counts on from them
    hold = {"to": None}  # after Small first, the small picture the full size holds its broad shapes to
    hold_curve = schedule(cfg["small_hold_curve"]) if small and float(cfg["small_hold"] or 0) else None
    hold_size = max(8, min(side_x, side_y) // 24)  # shapes broader than this are held: 32 pixels at 1280x768
    def inpainter(picture, hard, asked):
        """Fix's Inpaint: the picture, 0 to 1, with the area repainted by the Inpaint model the page chose, and the
        soft mask it went back in through. The model is kept for the rest of the render, shared with a handoff that
        uses the same file, so later inpaints and tries don't load it again, and let go when the render ends unless
        the handoff uses it. Disco waits in RAM while it paints if the two won't fit on the GPU together."""
        import handoff
        import inpaint
        path = asked["file"]
        painter = kept(("handoff", path, os.path.getmtime(path)), lambda: handoff.load(path, hooks.status, device), brief=True)
        image, size = imaging.to_pil(picture), (picture.shape[-1], picture.shape[-2])
        resting = [model, secondary_model, lpips_model, *(m.visual for m in clip_models), experiment.partner if experiment else None]
        release()
        with handoff.Parked(painter, resting, device, *painter.inpaint_size(*size), painted=True):
            painted, paste = inpaint.paint(painter, image, imaging.mask_to_pil(hard), asked.get("prompt", ""), asked["strength"],
                                    asked["steps"], asked["guidance"], asked.get("seed", seed + eraser.count), hooks.status,
                                    getattr(hooks, "stopped", lambda: False),  # a new try paints anew
                                    asked.get("context", inpaint.CONTEXT), asked.get("grow", inpaint.GROW), asked.get("blend", inpaint.BLEND))
        return imaging.from_pil(painted, picture), imaging.mask_from_pil(paste, picture.device)

    # Fix: what was painted over during a pause, erased, inpainted or cloned, and repainted by the steps left.
    eraser = erase.Erase(hooks.job_dir, device, diffusion, hooks.status, inpainter, hooks.tried, kept) if getattr(hooks, "job_dir", None) else None
    if eraser:
        hooks.while_paused = lambda: eraser.try_paint(hooks.pausing)  # inpaint tries, painted while the render waits

    def predict(x, t):
        """The model's own picture and noise for x at timestep t, without guidance, on x's device: for an erase."""
        tt = torch.tensor([t], device=device)
        with torch.no_grad():
            out = diffusion.p_mean_variance(one_pass or model, x.to(device), tt, clip_denoised=cfg["clip_denoised"], model_kwargs={})
            eps = diffusion._predict_eps_from_xstart(x.to(device), tt, out["pred_xstart"])
        return out["pred_xstart"].to(x.device), eps.to(x.device)

    def forget_erased():
        """After an erase, what the render is held to forgets the thing too: the start image Init scale compares
        against, Small first's picture, the Controls' picture and a ghost. Otherwise they would keep pulling it back."""
        nonlocal init
        eraser.erased = False
        init = eraser.cover(init)
        pull["to"] = eraser.cover(pull["to"])
        hold["to"] = eraser.cover(hold["to"])
        if steer and steer.source is not None:
            steer.set_source(eraser.cover(steer.source, unit=True))
        ghost = experiment.ghost if experiment else None
        if ghost and ghost.ghost is not None:  # a Ghost image or a sketch made first; Live ghost redraws from the render
            ghost.ghost = eraser.cover(ghost.ghost, unit=True)
            if ghost.shapes is not None:
                with torch.no_grad():
                    ghost.shapes = ghost.relations(ghost.ghost)

    def hold_layout(x, estimate, i):
        """Hold the small layout: after the step at index i, moves the render's broad shapes and colors toward the
        small picture, as a ghost's light and dark is moved, and leaves the detail free."""
        if hold["to"] is None or hold_curve is None or i == 0:
            return x
        at = schedule_at(i, diffusion.num_timesteps)
        strength = float(cfg["small_hold"]) * hold_curve[at]
        if not strength:
            return x
        scale = math.sqrt(float(diffusion.alphas_cumprod_prev[i]))
        return x + (strength * scale * lab.low_pass(hold["to"] - estimate.detach().float(), hold_size)).to(x.dtype)
    one_pass = OnePassUNet(model) if cfg["diffusion_sampling_mode"] == "ddim" and not secondary else None

    def cond_fn(x, t, y=None):
        cur_t = progress["t"]
        take_regions()  # before the step's graph is built, as it may bring the text encoders back for a moment
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
            view = view_at(at)  # the views schedules' point, which Fit views and Shapes share can move
            guidance = cfg["clip_guidance_scale"] * experiment.guidance[at] if experiment else cfg["clip_guidance_scale"]
            for stat in stats:
                share = judge.model_weight(stat["model"], at) if judge else 1.0
                if not share:
                    continue  # a checker with no say at this step costs no time
                own = regions["embeds"][stat["model"]] if regions["boxes"] else None
                # with regions, their prompts join the main ones, and each cut weighs them by how much it covers
                full = stat if own is None else {**stat, "embeds": torch.cat([stat["embeds"], own]),
                                                 "weights": F.pad(stat["weights"], (0, len(own)))}
                for _ in range(cfg["cutn_batches"]):
                    faces = steer.face_crops(x_in, at) if steer else []  # Face detail: close-ups on each face
                    cuts = MakeCutoutsDango(stat["model"].visual.input_resolution, int(overview[view]), int(innercut[view]),
                                            ic_pow[view], icgray_p[view], cfg["skip_augs"], cfg["exact_repeat"],
                                            experiment.evidence if experiment else None, faces)
                    whole = place(x_in).add(1).div(2) if place and int(overview[view]) else None
                    batch = normalize(cuts(x_in.add(1).div(2), whole))
                    total = int(overview[view]) + int(innercut[view]) + len(faces)
                    chunk = memory.clip_chunk(stat["model"]) * n
                    rows = judge.prompt_weights(stat, total, int(overview[view]), at) if judge else None
                    if own is not None:
                        rows = region_rows(stat["weights"] if rows is None else rows, cuts.boxes, int(overview[view]), total,
                                           x_in.shape[2], x_in.shape[3])
                    weights = full["weights"] if rows is None else rows[:, None, :]  # per cut when prompts are routed or regions set
                    layout = judge is not None and judge.layout_on(at)
                    if chunk >= batch.shape[0]:
                        image_embeds = stat["model"].encode_image(batch).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), full["embeds"].unsqueeze(0))
                        dists = dists.view([total, n, -1])
                        if layout:
                            x_in_grad += share * judge.layout_grad(full, batch, dists, int(overview[view]), n, x_in, chunk, guidance,
                                                                   at, rows) / cfg["cutn_batches"]
                        losses = dists.mul(weights).sum(2).mean(0)
                        x_in_grad += torch.autograd.grad(losses.sum() * guidance * share, x_in)[0] / cfg["cutn_batches"]
                    else:  # a few cuts at a time through a big CLIP, then one pass back through the cutting
                        leaf, cut_grad, parts = batch.detach().requires_grad_(), 0, []
                        for i in range(0, leaf.shape[0], chunk):
                            image_embeds = stat["model"].encode_image(leaf[i:i + chunk]).float()
                            part = spherical_dist_loss(image_embeds.unsqueeze(1), full["embeds"].unsqueeze(0)).view([-1, n, len(full["weights"])])
                            w = weights if rows is None else weights[i // n:i // n + part.shape[0]]
                            loss = part.mul(w).sum(2).sum(0).sum() / total
                            cut_grad = cut_grad + torch.autograd.grad(loss * guidance * share, leaf)[0]
                            parts.append(part.detach())
                        dists = torch.cat(parts)
                        if layout:
                            x_in_grad += share * judge.layout_grad(full, batch, dists, int(overview[view]), n, x_in, chunk, guidance,
                                                                   at, rows) / cfg["cutn_batches"]
                        x_in_grad += torch.autograd.grad(batch, x_in, cut_grad)[0] / cfg["cutn_batches"]
                    if cuts.evidence:
                        cuts.evidence.add(cuts.boxes, dists.mul(full["weights"]).sum(2)[int(overview[view]):, 0])
            if experiment:
                x_in_grad = experiment.shape_push(x_in_grad, at)
                if experiment.ghost:
                    x_in_grad = experiment.ghost.push(x_in, x_in_grad, at)
            if steer:
                x_in_grad = steer.push(x_in, x_in_grad, at)
            if judge:
                x_in_grad = judge.subject_push(x_in, x_in_grad, at, normalize)
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
                           "offload": offload, "range_scale_works": True, "width_height": [side_x, side_y], "final_size": list(final),
                           **({"small_size": list(small), "small_steps": small_steps} if small else {})}, f, indent=2)

    def save(image, batch):
        path = os.path.join(folder, f"{batch_name}({run})_{batch}.png")
        write_notes(again=regions["dirty"] and os.path.exists(os.path.join(folder, f"{batch_name}({run})_settings.txt")))
        regions["dirty"] = False
        image.save(path)
        return path

    def pausing():
        return getattr(hooks, "pausing", lambda: False)()

    def to_image(tensor):
        return TF.to_pil_image(tensor[0].add(1).div(2).clamp(0, 1).cpu())

    def denoise(start, width, height, skip, batch, mask=None, show=to_image, stop=None):
        """Runs one render and returns its finished image as a -1 to 1 tensor. With `stop`, it ends after that
        many steps and returns its picture at that point, still blurry."""
        nonlocal init
        init = pull["to"] if pull["set"] else start  # Init scale compares against this
        painting_tile = mask is not None
        common = dict(clip_denoised=cfg["clip_denoised"], model_kwargs={}, cond_fn=cond_fn, progress=False,
                      skip_timesteps=skip, init_image=start, randomize_class=cfg["randomize_class"])
        progress["t"] = diffusion.num_timesteps - skip - 1
        before, ahead["steps"] = ahead["steps"], 0
        total_steps = progress["t"] + 1 + before
        shape = (1, 3, height, width)
        common["noise"] = start_noise(shape, device)
        if cfg["diffusion_sampling_mode"] == "ddim":
            samples = diffusion.ddim_sample_loop_progressive(
                one_pass or model, shape, eta=cfg["eta"], transformation_fn=None if painting_tile else symmetry,
                transformation_percent=cfg["transformation_percent"], inpainting_mode=painting_tile, mask_inpaint=mask, **common)
        else:
            samples = diffusion.plms_sample_loop_progressive(model, shape, order=2, **common)
        for j, sample in enumerate(samples):
            progress["t"] -= 1
            last = progress["t"] == -1 or j + 1 == stop
            if j % cfg["display_rate"] == 0 or last or pausing():
                hooks.preview(show(sample["pred_xstart"]))
            if eraser:
                eraser.shown = sample["pred_xstart"]
            hooks.step(j + before, total_steps, batch, n_batches, seed,  # after the preview, so a pause shows this step
                       at=1 - (progress["t"] + 1) / diffusion.num_timesteps)
            if j + 1 == stop:
                break
            if hold["to"] is not None:  # the sampler reads the sample back for the next step
                sample["sample"] = hold_layout(sample["sample"], sample["pred_xstart"], progress["t"] + 1)
            if eraser and not painting_tile and eraser.step(sample["pred_xstart"]):
                forget_erased()
                sample["sample"] = eraser.restart(sample["sample"], progress["t"] + 1, predict)
            if steer:
                sample["sample"] = steer.pull(sample["sample"], sample["pred_xstart"], progress["t"] + 1, diffusion)
        release()
        return sample["pred_xstart"].detach()

    def denoise_lab(start, width, height, skip, batch, stop=None, show=to_image):
        """The normal ddim loop, with room for the experiments that work between steps. `stop` and `show` work as in denoise."""
        nonlocal init
        init = pull["to"] if pull["set"] else start
        before, ahead["steps"] = ahead["steps"], 0
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
        x = start_noise((1, 3, height, width), device)
        if x is None:
            x = torch.randn(1, 3, height, width, device=device)
        if skip or start is not None:
            base = torch.zeros_like(x) if start is None else start
            x = diffusion.q_sample(base, torch.tensor([first], device=device), x)
        image = torch.zeros_like(x)
        for j, i in enumerate(range(first, -1, -1)):
            progress["t"] = i
            if eraser:
                eraser.shown = image  # the last step's picture, which a pause here shows
            hooks.step(j + before, first + 1 + before, batch, n_batches, seed, at=1 - i / diffusion.num_timesteps)
            t = torch.tensor([i], device=device)
            at = schedule_at(i, diffusion.num_timesteps)
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
            x = hold_layout(x, image, i)
            if eraser and eraser.step(image):
                forget_erased()
                x = eraser.restart(x, i, predict)
            if steer:
                x = steer.pull(x, image, i, diffusion)
            if band:
                moved = experiment.share(helper.sketch(), image, band)
                image = image + moved
                x = x + math.sqrt(alpha_prev) * moved
                if experiment.band_path == "Blend":  # the noise estimate changes with the mix, so the step stays small
                    sigma = cfg["eta"] * math.sqrt((1 - alpha_prev) / (1 - alpha)) * math.sqrt(1 - alpha / alpha_prev)
                    x = x - math.sqrt(max(1 - alpha_prev - sigma ** 2, 0)) * math.sqrt(alpha / (1 - alpha)) * moved
            if judge and cfg["correct_steps"] and judge.correct_curve[at] and i > 0:
                # Langevin steps at the new noise level, so the picture settles there before the next step. With
                # Two painters they aim at both models' combined prediction, so the path samples both at once.
                progress["t"] = i - 1
                t_next = torch.tensor([i - 1], device=device)
                for _ in range(int(cfg["correct_steps"])):
                    with torch.no_grad():
                        mine = diffusion.ddim_sample(one_pass, x, t_next, clip_denoised=cfg["clip_denoised"],
                                                     cond_fn=cond_fn if cfg["correct_clip"] else None, model_kwargs={}, eta=0)["pred_xstart"]
                        if helper and band:
                            theirs = helper.denoise(mine, float(alpha_prev))
                            mine = mine + experiment.share(theirs, mine, band)
                        x = judge.corrector(x, mine, float(alpha_prev), float(cfg["correct_snr"]))
                image = mine.detach()
            if j % cfg["display_rate"] == 0 or not i or j + 1 == stop or pausing():
                hooks.preview(show(image))
            if j + 1 == stop:
                break
        if helper and needed and helper.z0 is not None and cfg["partner_keep_sketch"]:
            save_sketch(helper.sketch())
        release()
        return image

    def release():
        """The last step's UNet output keeps its whole graph alive until the next step, so let it go."""
        if one_pass:
            one_pass.leaf = one_pass.output = None

    def hand_over(start, skip, n, size, noise_seed, keep_as=None):
        """With a handoff model, it paints from `start` (a -1 to 1 tensor, or None for noise), and Disco takes
        its picture. At Handoff 100% the checkpoint finishes it and Disco repaints `skip`'s share, Denoise, as it
        would a start image. Below that it stops partway and Disco carries on from that point. Returns the new
        start and skip. `size` is the size painted when there is no start."""
        pull["set"], pull["to"] = False, None
        if handing is None or not hand_start:
            return start, skip
        release()
        # The checkpoint repaints Start denoise of a start, or with that at 0, the share Disco itself would.
        own = float(cfg.get("handoff_start_denoise") or 0) / 100
        repaints = own or (diffusion.num_timesteps - skip) / diffusion.num_timesteps
        picture = None if start is None else to_image(start)
        resting = [model, secondary_model, lpips_model, *(m.visual for m in clip_models),
                   experiment.partner if experiment else None]
        with handoff.Parked(handing, resting, device, *size):
            made = handing.paint(picture, repaints, n, *size, noise_seed)
        if made is None:
            return start, skip
        image, alpha = made
        if keep_as:
            image.save(keep_as)
        pull["set"], pull["to"] = True, start
        if float(cfg["handoff_share"]) < 100:  # it carries on from the point on its own path where the checkpoint stopped
            skip = handoff.disco_skip(diffusion.alphas_cumprod, alpha, skip)
        hooks.status(f"Disco carries on from step {skip + 1} of {cfg['steps']}")
        return TF.to_tensor(image).unsqueeze(0).mul(2).sub(1).to(device), skip

    def small_start(start, skip, n, paint):
        """Small first: Disco paints the start of the path at `small`, in the frame's shape, and stops at
        small_steps while the picture is still blurry. Enlarged, that picture is the start for the rest of the
        path at full size, where Disco adds the noise for that step as it does to any start image, so it
        carries on from the same point. Returns the new start and skip, like hand_over."""
        hold["to"] = None
        if not small_steps or skip >= small_steps:
            return start, skip
        width, height = small

        def sized(image, w, h):
            if image is None:
                return None
            return F.interpolate(image.float(), size=(h, w), mode="bicubic", antialias=True, align_corners=False)

        held = pull["to"] if pull["set"] else start  # what Init scale compares against at full size
        pull["set"], pull["to"] = True, sized(held, width, height)
        hooks.status(f"Painting steps {skip + 1} to {small_steps} of {cfg['steps']} at {width}x{height}")
        # The preview is enlarged to the frame, so it looks the same size all the way through.
        rough = paint(sized(start, width, height), width, height, skip, n, stop=small_steps - skip,
                      show=lambda image: to_image(sized(image, side_x, side_y)))
        pull["set"], pull["to"] = True, held
        ahead["steps"] = small_steps - skip
        carry_on = small_steps
        repaint = float(cfg["small_denoise"] or 0)
        if repaint > 0:  # the full size repaints its share of the small picture, never less than what is left
            carry_on = min(max(cfg["steps"] - round(cfg["steps"] * repaint / 100), 0), small_steps)
        hooks.status(f"Carrying on at {side_x}x{side_y} from step {carry_on + 1} of {cfg['steps']}")
        enlarged = sized(rough, side_x, side_y).clamp(-1, 1)
        if hold_curve is not None:
            hold["to"] = enlarged  # Hold the small layout pulls the full size toward this
        return enlarged, carry_on

    def finish(image, n, seed_value, keep_as=None):
        """Finish %: the handoff model repaints the end of Disco's finished picture to clean it up."""
        if handing is None or not hand_finish:
            return image
        release()
        if keep_as:
            image.save(keep_as)  # Disco's own picture, before the finish
        resting = [model, secondary_model, lpips_model, *(m.visual for m in clip_models),
                   experiment.partner if experiment else None]
        with handoff.Parked(handing, resting, device, *handing.finish_size(*image.size), painted=True):
            return handing.finish(image, float(cfg["handoff_finish"]) / 100, n, seed_value)

    def save_sketch(sketch):
        to_image(sketch).save(os.path.join(folder, f"{batch_name}({run})_partner.png"))

    def paint_tiles(start, skip, batch):
        """Takes every tile through every step together, on one picture held in RAM.

        Each step, every tile predicts its part and the overlaps are averaged before the picture moves
        on, so neighbors settle on the same image instead of being joined at the end. Each tile is an
        ordinary render as far as the GPU is concerned.
        """
        nonlocal place, init, tile_at
        size = (1, 3, final[1], final[0])
        noise = start_noise(size, "cpu")
        if noise is None:
            noise = torch.randn(size)
        first = diffusion.num_timesteps - skip - 1
        canvas = noise if skip == 0 and start is None else None
        if canvas is None:
            canvas = diffusion.q_sample(torch.zeros(size) if start is None else start, torch.tensor([first]), noise)
        latest = start

        def paint_tile(source, i, x, y, view):
            """One tile's guided prediction of the picture and of the noise, from `source` at timestep i."""
            nonlocal place, init, tile_at
            box = (..., slice(y, y + side_y), slice(x, x + side_x))
            tile = source[box].to(device)
            place = view and cfg["tile_whole_view"] and tiles.placer(*view, x, y, side_x, side_y, device)
            tile_at = (x, y)
            held = pull["to"] if pull["set"] else start
            init = None if held is None else held[box].to(device)  # Init scale compares against this
            t = torch.tensor([i], device=device)
            with torch.no_grad():
                out = diffusion.p_mean_variance(one_pass or model, tile, t, clip_denoised=cfg["clip_denoised"], model_kwargs={})
                pred = diffusion.condition_score(cond_fn, out, tile, t, model_kwargs={})["pred_xstart"]
                eps = diffusion._predict_eps_from_xstart(tile, t, pred)
                if not i:
                    pred = out["pred_xstart"]  # as in a normal render, the last image is the model's own
            return pred.cpu(), eps.cpu()

        def predict_tiles(source, t):
            """predict() over the whole picture a tile at a time, the overlaps blended as a step blends them. Tiles
            that no erase touches are left out, as only the erased area is used."""
            across = tiles.fades(final[0], side_x - overlap, grid, overlap)
            down = tiles.fades(final[1], side_y - overlap, grid, overlap)
            area = eraser.back
            pred_sum, eps_sum = torch.zeros(size), torch.zeros(size)
            for number, (x, y) in enumerate(spots):
                box = (..., slice(y, y + side_y), slice(x, x + side_x))
                if not area[box].any():
                    continue
                weight = down[number // grid][y:y + side_y, None] * across[number % grid][x:x + side_x]
                pred, eps = predict(source[box], t)
                pred_sum[box] += pred.cpu().float() * weight
                eps_sum[box] += eps.cpu().float() * weight
            return pred_sum, eps_sum

        def ddim(pred, eps, i):
            """One DDIM step on the whole picture, so its noise doesn't get averaged away in the overlaps."""
            alpha, alpha_prev = diffusion.alphas_cumprod[i], diffusion.alphas_cumprod_prev[i]
            sigma = cfg["eta"] * math.sqrt((1 - alpha_prev) / (1 - alpha)) * math.sqrt(1 - alpha / alpha_prev)
            sample = pred * math.sqrt(alpha_prev) + math.sqrt(1 - alpha_prev - sigma ** 2) * eps
            return sample + sigma * torch.randn(size) if i else sample

        for j, i in enumerate(range(first, -1, -1)):
            progress["t"] = i
            view = None if latest is None else tiles.small_copy(latest)
            pred_sum, eps_sum = torch.zeros(size), torch.zeros(size)
            across = tiles.fades(final[0], side_x - overlap, grid, overlap)
            down = tiles.fades(final[1], side_y - overlap, grid, overlap)
            for number, (x, y) in enumerate(spots):
                if eraser and not number:
                    eraser.shown = latest
                hooks.step(j * len(spots) + number, (first + 1) * len(spots), batch, n_batches, seed, can_hold=not number,
                           at=1 - i / diffusion.num_timesteps)
                if eraser and not number and latest is not None and eraser.step(latest):
                    # A pause holds here, on the last step's picture, which is what the page painted over, so an erase
                    # comes before this step paints anything. The canvas is at the last step's noise level, i + 1's.
                    start = eraser.cover(start)  # each tile's Init scale compares against its part of this
                    latest = eraser.cover(latest)
                    forget_erased()
                    canvas = eraser.restart(canvas, i + 1, predict_tiles)
                    view = tiles.small_copy(latest)
                    shown = to_image(latest)
                    shown.thumbnail((1024, 1024))
                    hooks.preview(shown)  # the erased picture at once, not a step later
                box = (..., slice(y, y + side_y), slice(x, x + side_x))
                weight = down[number // grid][y:y + side_y, None] * across[number % grid][x:x + side_x]
                pred, eps = paint_tile(canvas, i, x, y, view)
                pred_sum[box] += pred * weight
                eps_sum[box] += eps * weight
            place = tile_at = None
            canvas = ddim(pred_sum, eps_sum, i)
            latest = pred_sum.clamp(-1, 1)
            # A pause holds before the next step's first tile, so the page paints over this step's picture.
            if j % max(cfg["display_rate"] // len(spots), 1) == 0 or not i or pausing():
                preview = to_image(latest)
                preview.thumbnail((1024, 1024))  # saving a full-size preview takes longer than it's worth
                hooks.preview(preview)
        release()
        return latest

    def film():
        """Renders the frames of an animation, each from the last one moved by the camera or from a video frame."""
        nonlocal stats, init_scale, lpips_model, n_batches
        mode = cfg["animation_mode"]
        loop = cfg["loop_mode"] if cfg["loop_mode"] in ("Paint back to the start", "Ping-pong") else None
        write_notes(again=bool(resume))  # claims the run number, so a stopped animation keeps its frames
        frames_dir = os.path.join(folder, f"{batch_name}({run})_frames")
        os.makedirs(frames_dir, exist_ok=True)
        video = None
        if mode == "Video Input":
            hooks.status("Reading the video")
            video = animate.VideoInput(cfg["video_init_path"], max(int(cfg["extract_nth_frame"]), 1), *final,
                                       init_fit, size_start_image, int(cfg["max_frames"]), loop == "Ping-pong")
        total = min(int(cfg["max_frames"]), len(video.frames)) if video else int(cfg["max_frames"])
        # Seamless loop: the last `reach` frames are painted more and more like the frame they lead back to. With
        # ping-pong that is the frame at the same point on the way out, so it must already be painted.
        reach = min(max(int(cfg["loop_frames"]), 1), (total - 1) // 2 if loop == "Ping-pong" else total - 1) if loop else 0
        circle = None  # with ping-pong and one shared noise, the noise goes once around a circle between two draws
        n_batches = total
        camera = animate.Camera(cfg, total, device, hooks.status, depth_model) if mode in ("2D", "3D") else None
        skips = [min(round(cfg["steps"] * share / 100), cfg["steps"] - 1) for share in keyframes.skips(cfg, total)]
        if cfg["frames_scale"] and lpips_model is None:
            lpips_model = load_lpips()
        whole = denoise_lab if experiment else denoise

        def paint(start, skip, n, hand=True):
            """One frame. With tiles, each frame is tiled like a still."""
            # before a handoff moves where Disco starts, so the views stay spread over the whole frame. A video's frames
            # and every frame after the first are clip frames; a 2D or 3D clip's first frame is its start image.
            fit_views(skip, frame=bool(video) or n > 0)
            if hand:  # the last frame's Small first is not this one's
                pull["set"], pull["to"] = False, None
            if hand and handing:
                start, skip = hand_over(start, skip, n, (side_x, side_y) if grid == 1 else final, frame_seed)
                set_seeds(frame_seed)  # each frame starts alike, whatever the handoff drew
            if grid == 1:
                start, skip = small_start(start, skip, n, whole)
                return whole(start, side_x, side_y, skip, n)
            return paint_tiles(None if start is None else start.float().cpu(), skip, n).to(device)

        turbo = max(int(cfg["turbo_steps"]), 1) if cfg["turbo_mode"] and camera else 1
        refresh = {at + int(cfg["prompt_fade"]) for at, _ in all_stats[1:]}  # a new prompt gets its own colors
        first, previous, saved = None, None, {}
        rendered_first = False  # the first frame's Render first pass already had its handoff
        between, painted_at = [], 0  # with Turbo, the frames moved on from the last repaint
        begin = 0

        def frame_at(n):
            return TF.to_tensor(Image.open(saved[n]).convert("RGB")).unsqueeze(0).mul(2).sub(1).to(device)

        if resume:  # pick up after the last frame on disk, as if the run had never stopped
            while os.path.isfile(os.path.join(frames_dir, f"{begin:05d}.png")):
                saved[begin] = os.path.join(frames_dir, f"{begin:05d}.png")
                begin += 1
            if not begin:
                raise ValueError("That clip has no frames to continue from.")
            if begin >= total:
                raise ValueError(f"That clip already has {begin} frames. Raise Frames to make it longer.")
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
                if video and loop == "Ping-pong" and cfg["video_init_seed_continuity"]:
                    # The same noise would paint the way back as a copy of the way out. Going round a circle, each
                    # frame's noise is close to the last one's, and the frame after the last is back at the first's.
                    sized = (1, 3, final[1], final[0]) if grid > 1 else (1, 3, side_y, side_x)
                    if circle is None:
                        circle = [torch.randn(sized, generator=torch.Generator().manual_seed(seed + k)) for k in (0, 1)]
                    turn = 2 * math.pi * n / total
                    given["noise"] = math.cos(turn) * circle[0] + math.sin(turn) * circle[1]
                set_seeds(frame_seed)
                if steer and (fixed_control is not None or video):
                    # the Control image, or else the clean video frame, never the start the last frame was mixed into
                    steer.set_source(fixed_control if fixed_control is not None else TF.to_tensor(video.frames[n]).unsqueeze(0))
                if video:
                    shown = TF.to_tensor(video.frames[n]).unsqueeze(0).mul(2).sub(1).to(device)
                    start = shown
                    if previous is None and cfg["video_blend"]:
                        # The first frame builds on a first pass of itself, as later frames build on the last
                        # one. Without this it stays close to the video while the rest of the clip doesn't.
                        hooks.status("Painting a first pass of the first frame")
                        previous = paint(shown, skip_steps, n)  # a full repaint from Denoise, as a still would be
                    if previous is not None and cfg["video_blend"]:
                        last = previous
                        if cfg["video_flow"] and n:  # the last frame's paint travels with what moved in the video
                            last = animate.follow_flow(previous, animate.video_flow(video.frames[n - 1], video.frames[n]), cfg)
                        # The carried paint is repainted every frame, and each pass adds contrast and color. Held to the
                        # first frame's light and color, as camera moves are, it can't compound into a burned-out clip.
                        last = animate.hold_colors(last, first, cfg["color_hold"])
                        start = shown + cfg["video_blend"] * (last - shown)
                    skip = skips[n]
                elif n == 0:
                    shown = start = TF.to_tensor(source).unsqueeze(0).mul(2).sub(1).to(device) if start_image else None
                    skip = skip_steps  # from a start image or noise, with tiles or without
                    if grid > 1 and not start_image and cfg["tile_start"] == "Render first":
                        hooks.status("Painting the first frame at tile size")
                        fit_views(skip_steps, frame=bool(video))
                        rough, rough_skip = hand_over(None, skip_steps, n, (side_x, side_y), frame_seed)
                        set_seeds(frame_seed)
                        rough = to_image(denoise(rough, side_x, side_y, rough_skip, n)).resize(final, Image.LANCZOS)
                        shown = start = TF.to_tensor(rough).unsqueeze(0).mul(2).sub(1).to(device)
                        skip, rendered_first = tile_skip, True
                else:
                    moved = animate.sharpen(camera.move(previous, n), cfg["frame_sharpen"])
                    shown = start = animate.hold_colors(moved, first, cfg["color_hold"])
                    skip = skips[n]
                if reach and n >= total - reach:
                    back_to = (n - (total - reach) + 1) / (reach + 1)
                    target = frame_at(total - n if loop == "Ping-pong" else 0)
                    hooks.status(f"Painting back to the start, {round(100 * back_to)}% of the way")
                    start = start + back_to * (target - start)
                init_scale = 0 if start is None else cfg["frames_scale"] if (n or video) else cfg["init_scale"]
                if steer and fixed_control is None and not video:  # a camera move follows the frame it starts from
                    steer.set_source(None if shown is None else shown.add(1).div(2))
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
                    saved = animate.add_between(saved, extra, os.path.join(frames_dir, "between"), device, hooks.status,
                                                wrap=bool(loop) and len(saved) == total)
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
    # Tile denoise is only for Render first, where the tiles repaint the picture its first pass made.
    tile_skip = min(round(cfg["steps"] * (1 - cfg["tile_denoise"])), cfg["steps"] - 1)
    if animating:
        film()
        return
    for batch in range(n_batches):
        gc.collect()
        torch.cuda.empty_cache()
        set_seeds(seed + batch)  # each image can be made again on its own
        start = TF.to_tensor(source).unsqueeze(0).mul(2).sub(1).to(device) if start_image else None
        if steer:
            steer.set_source(fixed_control if fixed_control is not None else start.add(1).div(2) if start is not None else None)
        handoff_picture = cfg["handoff_keep"] and os.path.join(folder, f"{batch_name}({run})_{batch}_handoff.png")
        if grid == 1:
            paint = denoise_lab if experiment else denoise
            fit_views(skip_steps)
            start, skip = hand_over(start, skip_steps, batch, (side_x, side_y), seed + batch, handoff_picture)
            set_seeds(seed + batch)
            start, skip = small_start(start, skip, batch, paint)
            image = to_image(paint(start, side_x, side_y, skip, batch))
        else:
            if not start_image and cfg["tile_start"] == "Render first":
                fit_views(skip_steps)
                start, skip = hand_over(None, skip_steps, batch, (side_x, side_y), seed + batch, handoff_picture)
                set_seeds(seed + batch)
                first = to_image(denoise(start, side_x, side_y, skip, batch))
                start, skip = TF.to_tensor(first.resize(final, Image.LANCZOS)).unsqueeze(0).mul(2).sub(1), tile_skip
                fit_views(tile_skip)
            else:
                fit_views(skip_steps)
                start, skip = hand_over(start, skip_steps, batch, final, seed + batch, handoff_picture)
                set_seeds(seed + batch)
            image = to_image(paint_tiles(None if start is None else start.cpu(), skip, batch))
        disco_own = cfg["handoff_keep"] and os.path.join(folder, f"{batch_name}({run})_{batch}_disco.png")
        image = finish(image, batch, seed + batch, disco_own)
        hooks.image(save(image, batch))
