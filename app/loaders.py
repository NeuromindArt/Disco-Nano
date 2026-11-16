"""Loading the render's models onto the GPU, and keeping them in RAM between renders.

Derived from Disco Diffusion, which is MIT licensed. See CREDITS.md and LICENSES/disco-diffusion.txt for its authors.
"""
import gc
import traceback

import clip
import torch
from guided_diffusion.script_util import create_gaussian_diffusion, create_model_and_diffusion, model_and_diffusion_defaults

import critic
import memory
import models
import paths
from secondary import SecondaryDiffusionImageNet2


device = torch.device("cuda")
CLIP_NAMES = {"ViTB32": "ViT-B/32", "ViTB16": "ViT-B/16", "ViTL14": "ViT-L/14", "RN50": "RN50", "RN101": "RN101",
              "RN50x4": "RN50x4", "RN50x16": "RN50x16", "RN50x64": "RN50x64"}


# ---- models kept between renders ----
# The render worker stays open between renders, and keeps the models it loaded in RAM for the next one. Each
# render brings what it needs to the GPU, and put_away sends it all back to RAM when the render ends, so a
# waiting worker holds no VRAM beyond its CUDA context. Closing the app ends the worker and frees it all. Models the
# last render didn't use are let go, so switching checkpoints or settings doesn't pile up copies in RAM. A model only
# Fix's Inpaint used is let go too: it's needed now and then, and holds many GB that the next render may want.
_kept = {}
_used = set()  # keys asked for during this render, to keep for the next one
_after_render = []  # undoes what a render changed in kept models or shared code, such as hooks


def after_render(undo):
    """Runs `undo` when the render ends, however it ends, to take back what it changed in a kept model or shared code."""
    _after_render.append(undo)


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
