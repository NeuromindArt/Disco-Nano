"""How much VRAM and RAM a render will take, part by part, and whether it needs Offload.

The numbers were measured on an RTX 3070 8GB. VRAM is the most the render's tensors held at once, plus a
small margin for PyTorch's allocator. RAM is what the render process committed, which Windows must find in
RAM or the page file. Each part grows the same way on any card, so the plan holds elsewhere too; only what
is free differs.
"""
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
# What each checker adds over the default three, or saves when it is off. Each one's working data is small next to
# the UNet's, so only the fixed part counts. On 8GB, adding RN50x16 by itself runs out of memory at 512x512.
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
# Measured at 512x512 and 768x512 with the same result, so they don't grow with the frame.
LAYOUT = {"All": ("Layout critic", 0.89, 0), "Whole-image": ("Layout critic, whole-image views", 0.65, 0)}
ROBUST = {"FARE4 B/16": ("Robust checker", 0.16, 0.6), "FARE2 L/14": ("Robust checker", 0.57, 1.7),
          "FARE4 L/14": ("Robust checker", 0.57, 1.7)}
# The controls' finders run once on the control picture, before the steps, so they add nothing to the peak.
FINDERS = {"pose_control": 0.24, "person_control": 0.19}  # RAM, as their files
FACES_RAM_GB = 0.11
# A handoff checkpoint waiting in RAM beside the render. Loading SDXL reads it in 32 bits before storing it in 8, which
# peaks higher, but that comes before Disco's models load, so it only counts where it's more than the two together.
# SDXL waiting was measured beside a 1280x768 render with Offload, which peaked at 28GB in all.
# Z-Image Turbo Q5 GGUF, Flux dev Q4 GGUF, Flux.2 Klein 4B Q8 GGUF and Klein Base 9B Q4 GGUF with their fp8 encoders,
# measured the same way: 28, 31, 25.6 and 33.4GB in all. They load as they are stored, with no higher peak, and stream
# from RAM, so they add nothing on the GPU between steps.
HANDOFF_RAM_GB = {"sd15": 4.3, "sdxl": 12.0, "zimage": 12.0, "flux": 15.0, "flux2": 10.0, "flux2_9b": 17.5}
HANDOFF_LOAD_GB = {"sd15": 4.3, "sdxl": 19.5, "zimage": 12.0, "flux": 15.0, "flux2": 10.0, "flux2_9b": 17.5}
HANDOFF_VRAM_GB = {"sd15": 0.0, "sdxl": 0.16, "zimage": 0.0, "flux": 0.0, "flux2": 0.0, "flux2_9b": 0.0}
HANDOFF_NAMES = {"sd15": "SD 1.5", "sdxl": "SDXL", "zimage": "Z-Image", "flux": "Flux", "flux2": "Flux.2 Klein 4B",
                 "flux2_9b": "Flux.2 Klein 9B"}


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
    if rules.on(cfg, "depth_target") or cfg["animation_mode"] == "3D" or float(cfg.get("depth_control") or 0) \
            or float(cfg.get("face_shape") or 0):
        extras.append(DEPTH)
    clip = cfg["animation_mode"] != "None"
    if (cfg.get("init_image") and cfg["init_scale"]) or (clip and float(cfg.get("frames_scale") or 0)):
        extras.append(LPIPS)
    if float(cfg.get("layout_critic") or 0):
        extras.append(LAYOUT.get(cfg.get("layout_cuts"), LAYOUT["All"]))
    if cfg.get("robust_critic") in ROBUST:
        extras.append(ROBUST[cfg["robust_critic"]])
    finders = sum(gb for name, gb in FINDERS.items() if float(cfg.get(name) or 0))
    if any(float(cfg.get(n) or 0) for n in ("face_control", "face_shape", "face_detail")):
        finders += FACES_RAM_GB
    if finders:
        extras.append(("Control finders", 0, finders))
    hand = str(cfg.get("handoff_file") or "")
    if any(setting_list.handoff_parts(cfg)):
        about = describe_checkpoint(hand) or {}
        kind = about.get("kind") + ("_9b" if about.get("size") == "9b" else "") if about.get("kind") else "sd15"
        kind = kind if kind in HANDOFF_RAM_GB else "sd15"
        extras.append(("Handoff model, between its steps", HANDOFF_VRAM_GB[kind], 0))
        ram_extras.append({"kind": "extra", "label": f"Handoff model ({HANDOFF_NAMES[kind]}), waiting in RAM",
                           "gb": HANDOFF_RAM_GB[kind], "loading": HANDOFF_LOAD_GB[kind]})
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
    for part in ram_extras:  # loading comes before the render's own models, so it counts only past them
        part["gb"] = max(part["gb"], part.pop("loading", 0) - render_ram)
    ram_parts += ram_extras

    return {
        "vram": {"total": gpu["total_gb"], "other": gpu["used_gb"],
                 "parts": [{"kind": "models", "label": label, "gb": fixed}, {"kind": "work", "label": work_label, "gb": work}, *extra_parts]},
        "ram": {"total": ram["total_gb"], "other": ram["used_gb"], "parts": ram_parts},
        "offload": offload,
        "too_big": need > room,
    }


# SD 1.5 and SDXL models made for few steps, such as SDXL Lightning, are told apart by these words in their names; the
# weights look the same. Z-Image and Flux say so through their variant. They start with the steps and CFG they want,
# and with their steps spaced as they expect.
FEW_STEPS = ("lightning", "turbo", "hyper", "lcm", "schnell")
FEW_STEP_DEFAULTS = {"handoff_steps": 8, "handoff_guidance": 1.5, "handoff_few_steps": True}
_kinds = {}  # what describe_checkpoint found, by path, change time and size


def describe_checkpoint(path):
    """{kind, inpaint_only for a model made only to inpaint, few_steps for one made for few steps, and the defaults it
    starts with, and for Z-Image and Flux their variant and label}, or None."""
    import flowfiles
    try:
        stat = os.stat(path)
        key = (path, stat.st_mtime, stat.st_size)
        if key not in _kinds:
            _kinds[key] = None
            if path.lower().endswith(".gguf"):
                _kinds[key] = flowfiles.describe(path)
            else:
                names, _ = flowfiles.safetensors_entries(path)
                unet = any(k.startswith("model.diffusion_model.input_blocks.") for k in names)
                # An inpainting checkpoint's first layer takes the picture and mask beside the noise: 9 channels.
                made = {"inpaint_only": True} if names.get("model.diffusion_model.input_blocks.0.0.weight", {}).get("shape", [0, 0])[1] == 9 else {}
                if any(word in path.lower() for word in FEW_STEPS):
                    made.update(few_steps=True, defaults=FEW_STEP_DEFAULTS)
                if unet and any(k.startswith("conditioner.embedders.1.") for k in names):
                    _kinds[key] = {"kind": "sdxl", **made}
                elif unet and any(k.startswith("cond_stage_model.transformer.") for k in names):
                    _kinds[key] = {"kind": "sd15", **made}
                else:
                    _kinds[key] = flowfiles.describe(path)
        return _kinds[key]
    except (OSError, ValueError):
        return None
