"""How much VRAM and RAM a render will take, part by part, and whether it needs Offload.

The numbers were measured on an RTX 3070 8GB. VRAM is the most the render's tensors held at once, plus a
small margin for PyTorch's allocator. RAM is what the render process committed, which Windows must find in
RAM or the page file. Each part grows the same way on any card, so the plan holds elsewhere too; only what
is free differs.
"""
import json
import os

import models
import rules
import settings as setting_list
from settings import DEFAULTS

KEEP_GB = 0.75  # left free for the desktop; see memory.VRAM_HEADROOM_GB
CONTEXT_GB = 0.2  # CUDA's own kernels, outside what PyTorch counts
MARGIN_GB = 0.15  # the allocator's slack over the tensors it holds

# Disco and the three default checkers: fixed, and the working data per megapixel of the frame.
BASE_GB, WORK_PER_MPX = 2.67, 5.62
OFFLOAD_BASE_GB, OFFLOAD_WORK_PER_MPX = 2.53, 2.27  # the rest of the working data waits in RAM
SECONDARY_BASE_GB, SECONDARY_WORK_PER_MPX = 2.75, 2.42  # the small model skips the big backward pass
SMALL_MODEL_SAVING_GB = 1.9  # the style models are smaller than the main one
# What each checker adds over the default three, or saves when it is off. Each one's working data is small
# next to the UNet's, so only the fixed part counts. On 8GB, adding RN50x16 by itself
# runs out of memory at 512x512.
CHECKERS = {"ViTB32": 0.35, "ViTB16": 0.5, "RN50": 0.43, "ViTL14": 0.58, "RN101": 0.23, "RN50x4": 1.38, "RN50x16": 2.3, "RN50x64": 4.5}
CHECKER_RAM = {"ViTL14": 1.5, "RN101": 1.9, "RN50x4": 3.8, "RN50x16": 6.0, "RN50x64": 12.0}
DEFAULT_CHECKERS = ("ViTB32", "ViTB16", "RN50")
OFFLOAD_CHECKER_PER_MPX = {"ViTL14": 0.55}  # with Offload, ViT-L/14's cuts are a larger share of what stays
# Commit for the render process: fixed, and per megapixel. Offload adds little, because the frames it moves
# reuse memory the process already holds.
RAM_BASE_GB, RAM_PER_MPX = 6.3, 8.2
OFFLOAD_RAM_BASE_GB, OFFLOAD_RAM_PER_MPX = 7.1, 8.0
PARTNER = ("Tiny SD, the partner", 0.62, 1.2)  # label, VRAM, RAM
DEPTH = ("Depth Anything", 0.1, 0.6)
LPIPS = ("Start image strength", 0.05, 1.2)
HANDOFF_RAM_GB = {"sd15": 4.3, "sdxl": 19.5}  # loading SDXL reads it in 32 bits before storing it in 8
HANDOFF_VRAM_GB = {"sd15": 0.0, "sdxl": 0.16}


def plan(settings, gpu, ram):
    """gpu: {"used_gb", "total_gb"} with no render running. ram: {"used_gb", "total_gb"}.
    Returns the parts on each, whether Offload is needed, and whether it still won't fit."""
    cfg = {**DEFAULTS, **settings}
    width, height = cfg.get("width_height", [768, 512])
    mpx = (int(width) // 64 * 64) * (int(height) // 64 * 64) / 1e6
    secondary = bool(cfg["use_secondary_model"])

    delta = sum(gb * ((1 if cfg.get(n) else 0) - (1 if n in DEFAULT_CHECKERS else 0)) for n, gb in CHECKERS.items())
    if cfg["diffusion_model"] != models.MAIN:
        delta -= SMALL_MODEL_SAVING_GB
    shown = sum(bool(cfg.get(n)) for n in CHECKERS)
    label = f"Disco and {shown} prompt checker{'s' * (shown != 1)}"

    extras, ram_extras = [], []
    if setting_list.uses_partner(cfg):
        extras.append(PARTNER)
    if rules._on(cfg, "depth_target") or cfg["animation_mode"] == "3D":
        extras.append(DEPTH)
    if cfg.get("init_image") and cfg["init_scale"]:
        extras.append(LPIPS)
    hand = str(cfg.get("handoff_file") or "")
    if hand and float(cfg.get("handoff_share") or 0) > 0:
        kind = "sdxl" if checkpoint_kind(hand) == "sdxl" else "sd15"
        extras.append(("Handoff model, between its steps", HANDOFF_VRAM_GB[kind], 0))
        ram_extras.append({"kind": "extra", "label": f"Handoff model ({'SDXL' if kind == 'sdxl' else 'SD 1.5'}), waiting in RAM",
                           "gb": HANDOFF_RAM_GB[kind]})
    extra_parts = [{"kind": "extra", "label": name, "gb": gb} for name, gb, _ in extras if gb >= 0.05]
    extra_vram = sum(gb for _, gb, _ in extras)
    ram_extra = sum(r for _, _, r in extras) + sum(CHECKER_RAM.get(n, 0) for n in CHECKERS if cfg.get(n))

    room = gpu["total_gb"] - KEEP_GB - gpu["used_gb"]
    work_label = f"Working data for {width}×{height}"
    if secondary:
        fixed, work = SECONDARY_BASE_GB + delta, SECONDARY_WORK_PER_MPX * mpx
    else:
        fixed, work = BASE_GB + delta, WORK_PER_MPX * mpx
    fixed += CONTEXT_GB + MARGIN_GB
    offload = not secondary and fixed + work + extra_vram > room
    if offload:
        grow = OFFLOAD_WORK_PER_MPX + sum(g for n, g in OFFLOAD_CHECKER_PER_MPX.items() if cfg.get(n))
        fixed, work = OFFLOAD_BASE_GB + delta + CONTEXT_GB + MARGIN_GB, grow * mpx
        work_label += ", with the rest offloaded"
    need = fixed + work + extra_vram

    render_ram = (OFFLOAD_RAM_BASE_GB + OFFLOAD_RAM_PER_MPX * mpx if offload else RAM_BASE_GB + RAM_PER_MPX * mpx) + ram_extra
    ram_parts = []
    if offload:
        moved = min((WORK_PER_MPX - OFFLOAD_WORK_PER_MPX) * mpx, render_ram / 2)
        ram_parts.append({"kind": "models", "label": "This render", "gb": render_ram - moved})
        ram_parts.append({"kind": "offloaded", "label": "Offloaded working data", "gb": moved})
    else:
        ram_parts.append({"kind": "models", "label": "This render", "gb": render_ram})
    ram_parts += ram_extras

    return {
        "vram": {"total": gpu["total_gb"], "other": gpu["used_gb"], "keep": KEEP_GB,
                 "parts": [{"kind": "models", "label": label, "gb": fixed}, {"kind": "work", "label": work_label, "gb": work}, *extra_parts]},
        "ram": {"total": ram["total_gb"], "other": ram["used_gb"], "parts": ram_parts},
        "offload": offload,
        "too_big": need > room,
    }


_kinds = {}


def checkpoint_kind(path):
    """"sd15" or "sdxl" for a checkpoint Handoff can use, or None for anything else, such as Flux, SD 3 or SD 2.
    It reads the names in the file's header, which is quick and needs no torch. File size can't tell: some
    SD 1.5 files are over 5GB."""
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime, stat.st_size)
        if key not in _kinds:
            with open(path, "rb") as f:
                length = int.from_bytes(f.read(8), "little")
                names = json.loads(f.read(length)) if 0 < length < 200 * 2**20 else {}
            unet = any(k.startswith("model.diffusion_model.input_blocks.") for k in names)
            if unet and any(k.startswith("conditioner.embedders.1.") for k in names):
                _kinds[key] = "sdxl"
            elif unet and any(k.startswith("cond_stage_model.transformer.") for k in names):
                _kinds[key] = "sd15"
            else:
                _kinds[key] = None
        return _kinds[key]
    except (OSError, ValueError):
        return None
