"""Which settings can't be used with the others, and why. The app grays these out, and render() refuses them.

One place for every rule, so the page and the renderer agree.
"""
import os

import models
from settings import DEFAULTS, LABELS, small_first_on

# Experiments change the generation loop in ways tiles, symmetry, plms and the secondary model can't follow.
EXPERIMENTS = ["time_travel", "prompt_graft", "live_ghost", "depth_target", "two_painters", "borrow_attention", "partner_model",
               "detail_bands", "shaped_guidance", "gradient_momentum", "evidence_map", "ghost_image", "style_image", "frame_ghost",
               "route_prompts", "layout_critic", "robust_critic", "one_subject", "correct_steps"]
LOOP_CHANGES = ["tile_scale", "use_secondary_model", "use_horizontal_symmetry", "use_vertical_symmetry", "diffusion_sampling_mode"]
PARTNER_USERS = ["prompt_graft", "live_ghost", "two_painters", "borrow_attention"]
# Sized to the frame when a render starts, so they can't follow Small first's start at another size.
FRAME_SIZED = PARTNER_USERS + ["depth_target", "ghost_image", "style_image", "frame_ghost"]
# Controls that follow a control picture. Face detail is one too, but finds faces in the render without one.
CONTROLS = ["edge_control", "depth_control", "pose_control", "person_control", "face_control", "face_shape"]
RESNETS = ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64"]
VITS = ["ViTB32", "ViTB16", "ViTL14"]
CHECKERS = RESNETS + VITS  # never reported as conflicts: they are blocked only to stop the last one going
# Strengths of something else. Grayed out with it, but never a conflict of their own.
PARAMETERS = {"style_strength", "ghost_structure", "ghost_crops", "handoff_keep", "n_batches", "partner_model", "subject_words"}
# Settings whose off value is a word.
OFF_WORDS = {"live_ghost", "route_prompts", "robust_critic"}


def _on(cfg, name):
    """True when a setting is doing something, as opposed to sitting at its off value."""
    value = cfg.get(name, DEFAULTS.get(name))
    if name == "tile_scale":
        return value not in (None, "", "Off")
    if name == "diffusion_sampling_mode":
        return value == "plms"
    if name in OFF_WORDS:
        return value not in (None, "", "Off")
    if name == "partner_model":
        return False  # a choice, not a switch: it is only in the way when the partner is used
    if name == "small_first":
        return small_first_on(cfg)
    if name == "frame_ghost":
        return bool(value) and cfg.get("animation_mode", "None") != "None"
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def regions_on(cfg):
    """True when the render has regional prompts: boxes with prompts of their own."""
    regions = cfg.get("regions")
    return isinstance(regions, dict) and any(str(b.get("text", "")).strip() for b in regions.get("boxes", []) if isinstance(b, dict))


def _names(names):
    shown = [LABELS.get(n, n) if n != "diffusion_sampling_mode" else "plms" for n in names]
    return shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]


def _have(name):
    return os.path.exists(models.path_of(name))


def blocked(cfg, downloaded=_have):
    """{setting: reason} for every setting the others rule out. A setting can be listed while it is on,
    when two settings that are both on conflict; conflicts() reports those."""
    cfg = {**DEFAULTS, **cfg}
    out = {}

    def block(names, reason):
        for name in names:
            out.setdefault(name, reason)

    animating = cfg["animation_mode"] != "None"
    loop = [n for n in LOOP_CHANGES if _on(cfg, n)]
    trying = [n for n in EXPERIMENTS if _on(cfg, n)]
    if loop:
        block(EXPERIMENTS, f"Experiments can't run with {_names(loop)}.")
    if trying:
        block(LOOP_CHANGES, f"Can't run with {_names(trying)}. Turn the experiments off first.")
    if _on(cfg, "tile_scale"):
        block(["diffusion_sampling_mode", "use_horizontal_symmetry", "use_vertical_symmetry"], "Tiles always use ddim, with no mirroring.")
    if _on(cfg, "diffusion_sampling_mode"):
        block(["use_horizontal_symmetry", "use_vertical_symmetry", "tile_scale"], "Needs the ddim sampler.")
    if _on(cfg, "use_horizontal_symmetry") or _on(cfg, "use_vertical_symmetry"):
        block(["diffusion_sampling_mode", "tile_scale"], "Mirroring needs ddim and no tiles.")
    sized = [n for n in FRAME_SIZED if _on(cfg, n)]
    if _on(cfg, "small_first"):
        block(FRAME_SIZED, "Can't run with Small first %, which paints the start at another size.")
        block(["tile_scale"], "Can't run with Small first %. Tiles have their own small start, Render first.")
    if sized:
        block(["small_first"], f"Can't run with {_names(sized)}, which are sized to the frame.")
    if _on(cfg, "tile_scale"):
        block(["small_first"], "Tiles have their own small start: Render first, under Text start.")
    if regions_on(cfg):
        block(["tile_scale"], "Can't run with Regional prompts. Turn them off first.")
    steering = [n for n in CONTROLS + ["face_detail"] if _on(cfg, n)]
    if _on(cfg, "tile_scale"):
        block(CONTROLS + ["face_detail"], "Controls can't run with Tiles.")
    if steering:
        block(["tile_scale"], f"Can't run with {_names(steering)}.")
    if not ((str(cfg.get("control_image") or "").strip() and not animating) or str(cfg.get("init_image") or "").strip() or animating):
        block(CONTROLS, "Needs a Control image or a start image to follow.")

    partner_named = cfg["partner_model"]
    if partner_named == "Tiny SD" and not downloaded(models.PARTNER):
        block(PARTNER_USERS, "Get Tiny SD from Models in All settings first.")
    if partner_named == "SD 1.5 file" and not os.path.isfile(str(cfg["partner_file"] or "")):
        block(PARTNER_USERS, "Set SD 1.5 file under Experimental in All settings first.")
    if partner_named != "Tiny SD":
        block(["prompt_graft"], "Needs Tiny SD as the partner model.")
    if cfg["diffusion_model"] != models.MAIN:
        block(["prompt_graft"], "Works only with the 512x512 model.")
    if _on(cfg, "prompt_graft"):
        block(["partner_model"], "Prompt graft needs Tiny SD.")
    robust = models.ROBUST_CHOICES.get(cfg["robust_critic"])
    if robust and not downloaded(robust[0]):
        block(["robust_critic"], f"Get {models.LABELS[robust[0]]} from Models in All settings first.")
    if not downloaded(models.DEPTH):
        block(["depth_target", "depth_control", "face_shape"], "Get Depth Anything from Models in All settings first.")
    elif not (_on(cfg, "live_ghost") or _on(cfg, "ghost_image") or (animating and cfg["frame_ghost"])):
        block(["depth_target"], "Needs Live ghost, a Ghost image or Last frame as ghost.")
    resnets = [n for n in RESNETS if cfg[n]]
    vits = [n for n in VITS if cfg[n]]
    if not resnets:
        block(["style_image", "style_strength"], "Needs a ResNet checker, such as RN50.")
    if not vits:
        block(["ghost_structure", "ghost_crops", "one_subject"], "Needs a ViT checker.")
    elif _on(cfg, "one_subject") and len(vits) == 1:
        block(vits, "One subject needs a ViT checker.")
    if _on(cfg, "style_image") and len(resnets) == 1:
        block(resnets, "Style image needs a ResNet checker.")
    if _on(cfg, "ghost_image") and (cfg["ghost_structure"] or cfg["ghost_crops"]):
        if not vits:
            block(["ghost_image"], "Ghost shapes and Ghost areas need a ViT checker.")
        elif len(vits) == 1:
            block(vits, "Ghost shapes and Ghost areas need a ViT checker.")
    if len(resnets) + len(vits) == 1:
        block(resnets + vits, "At least one checker is needed.")
    if not downloaded("secondary"):
        block(["use_secondary_model"], "Get the secondary model from Models in All settings first.")
    if animating:
        block(["handoff_keep", "n_batches", "handoff_finish"], "Not for animation.")
    if cfg["animation_mode"] == "3D" and not downloaded(models.DEPTH):
        block(["animation_mode"], "3D needs Depth Anything. Get it from Models in All settings.")
    return out


def clashing(cfg, downloaded=_have, ruled=None):
    """The settings that are on although something else rules them out."""
    ruled = blocked(cfg, downloaded) if ruled is None else ruled
    full = {**DEFAULTS, **cfg}
    return [name for name in ruled
            if (name == "animation_mode" and full["animation_mode"] == "3D")
            or (name != "animation_mode" and name not in PARAMETERS and name not in CHECKERS and _on(full, name))]


def conflicts(cfg, downloaded=_have, ruled=None):
    """A message for each setting that is on although something else rules it out."""
    ruled = blocked(cfg, downloaded) if ruled is None else ruled
    found = [ruled[n] if n == "animation_mode" else f"{LABELS.get(n, n)}: {ruled[n]}" for n in clashing(cfg, downloaded, ruled)]
    return list(dict.fromkeys(found))
