"""Inpaint: a model of your own paints something into a marked part of the picture, during a render or on a start
image.

Any model Handoff takes works: SD 1.5, SDXL, Z-Image, Flux.1 or Flux.2 Klein. It is set apart from the handoff model,
under Inpaint in Settings, with its own steps and guidance, and loads through the same code, so one file used for
both loads once. diffusers' inpainting pipeline for the model paints the area, putting back the rest of the picture
at every step's noise level, so the two meet without a seam.

The model paints a crop around the area, with as much again of the picture on each side to see what surrounds it,
at the size it works best at, and the area goes back into the picture with a soft edge. Late in a render Disco only
polishes what an area holds, and a model's inpaint is as detailed as the picture around it, so it keeps its detail
where an erase would come out soft.

What it starts from:
    fresh    noise: the model paints the area anew, from the prompt if there is one and from what is around it.
    rework   the picture as it is, painted over at `strength`, to change what's there.
    remove   LaMa's fill, painted over at `strength`: the area is cleared first, then painted.
"""
import numpy as np
from PIL import Image, ImageFilter

MODES = ("fresh", "rework", "remove")
FEW_STEPS = ("lightning", "turbo", "hyper", "lcm", "schnell")


def defaults(about, path):
    """Steps and guidance that suit a model, from what the checkpoint list knows about it: its kind, and for
    Z-Image and Flux their variant's own. Few-step SD and SDXL checkpoints are told apart by their names."""
    if about.get("defaults"):
        own = about["defaults"]
        return {"inpaint_steps": own["handoff_steps"], "inpaint_guidance": own["handoff_guidance"]}
    name = path.lower()
    if any(word in name for word in FEW_STEPS):
        return {"inpaint_steps": 8, "inpaint_guidance": 1.5}
    return {"inpaint_steps": 30, "inpaint_guidance": 7.0 if about.get("kind") == "sd15" else 6.0}


def crop_box(mask, width, height):
    """The box the model paints: the mask's bounds with as much again on each side, at least 64 pixels, inside the
    picture. `mask` is a numpy array, nonzero where to paint."""
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    grow = max(bottom - top, right - left, 64)
    return max(left - grow, 0), max(top - grow, 0), min(right + grow, width), min(bottom + grow, height)


def paint(model, picture, hard, soft, text, strength, steps, guidance, seed, status, stopped=lambda: False):
    """The picture, a PIL image, with the area repainted by `model` (a handoff.load model). `hard` and `soft` are
    PIL masks the picture's size: the model repaints `hard`, grown a little so the soft edge falls on its own paint,
    and the result goes back through `soft`."""
    model.status, model.stopped = status, stopped
    box = crop_box(np.asarray(hard) > 127, *picture.size)
    width, height = model.inpaint_size(box[2] - box[0], box[3] - box[1])
    grow = max(3, min(picture.size) // 64)
    area = hard.filter(ImageFilter.MaxFilter(grow * 2 + 1))  # the soft edge's outer half is the model's paint too
    status("Inpainting")
    painted = model.inpaint(picture.crop(box).resize((width, height), Image.LANCZOS),
                            area.crop(box).resize((width, height), Image.BILINEAR),
                            text.strip(), min(max(float(strength), 0.05), 1.0), max(int(steps), 1), float(guidance), seed)
    out = picture.copy()
    out.paste(painted.convert("RGB").resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS), box[:2], soft.crop(box))
    return out


def to_pil(picture):
    """A 1x3xHxW tensor of 0 to 1 as a PIL image."""
    return Image.fromarray((picture[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))


def from_pil(image, like):
    """A PIL image as a tensor like `like`, 1x3xHxW of 0 to 1."""
    import torch  # here, so the server can read defaults() without loading torch
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None].to(like.device, like.dtype)
