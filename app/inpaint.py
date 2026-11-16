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

from imaging import grow

MODES = ("fresh", "rework", "remove")
# As Crop and Stitch sets them: how much of the picture the model sees around an area, as a multiple of the area's
# size, how many pixels past the area it paints, and how wide the soft edge it goes back through is.
CONTEXT, GROW, BLEND = 1.2, 10, 20
LATENT = 16  # picture pixels in a model's latent pixel, at the most: 8, packed 2x2 in the flow models


def defaults(about):
    """Steps and guidance that suit a model, from what budget.describe_checkpoint() knows about it: its own, as for
    Z-Image, Flux and few-step models, or its kind's."""
    if about.get("defaults"):
        own = about["defaults"]
        return {"inpaint_steps": own["handoff_steps"], "inpaint_guidance": own["handoff_guidance"]}
    return {"inpaint_steps": 30, "inpaint_guidance": 7.0 if about.get("kind") == "sd15" else 6.0}


def crop_box(mask, width, height, context=CONTEXT):
    """The box the model paints, as Crop and Stitch makes it: the mask's bounds grown by `context` times their size,
    half of the extra on each side, inside the picture. `mask` is a numpy array, nonzero where to paint."""
    rows, cols = np.nonzero(mask.any(1))[0], np.nonzero(mask.any(0))[0]
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    across, down = round((right - left) * (context - 1) / 2), round((bottom - top) * (context - 1) / 2)
    return max(left - across, 0), max(top - down, 0), min(right + across, width), min(bottom + down, height)


def paint(model, picture, hard, text, strength, steps, guidance, seed, status, stopped=lambda: False, context=CONTEXT,
          grow_by=GROW, blend=BLEND):
    """The picture, a PIL image, with the area `hard` (a PIL mask the picture's size) repainted by `model`, a
    handoff.load model, and the soft mask it went back in through.

    As inpainting is done well elsewhere: the model paints the area and a band `grow_by` pixels wide around it, so
    what it draws can reach into its surroundings, a wake or a shadow, with the rest of the picture put back at every
    step. Everything it painted goes back in whole, and the soft edge, `blend` pixels wide, falls outside it, where
    the model was held to the picture, so nothing it drew is cut off and no seam is left. A model paints in latent
    pixels, LATENT pixels of its own size at the most, so the edge starts one of those out from the area. The crop
    is measured around all of the soft edge, so none of it is cut by the crop's border."""
    model.status, model.stopped = status, stopped
    half = int(blend) // 2
    area = grow(np.asarray(hard) > 127, int(grow_by))
    reach = LATENT + half
    for _ in range(3):  # a crop bigger than the model paints at has bigger latent pixels, so a wider edge
        box = crop_box(grow(area, reach + half), *picture.size, context)
        width, height = model.inpaint_size(box[2] - box[0], box[3] - box[1])
        needed = round(LATENT * (box[2] - box[0]) / width) + half
        if needed <= reach:
            break
        reach = needed
    else:
        box = crop_box(grow(area, reach + half), *picture.size, context)
        width, height = model.inpaint_size(box[2] - box[0], box[3] - box[1])
    paste = Image.fromarray(grow(area, reach).astype(np.uint8) * 255)
    if blend > 0:  # a Gaussian fades over about four of its sigmas
        paste = paste.filter(ImageFilter.GaussianBlur(blend / 4))
    status("Inpainting")
    painted = model.inpaint(picture.crop(box).resize((width, height), Image.LANCZOS),
                            Image.fromarray(area.astype(np.uint8) * 255).crop(box).resize((width, height), Image.BILINEAR),
                            text.strip(), min(max(float(strength), 0.05), 1.0), max(int(steps), 1), float(guidance), seed)
    out = picture.copy()
    out.paste(painted.convert("RGB").resize((box[2] - box[0], box[3] - box[1]), Image.LANCZOS), box[:2], paste.crop(box))
    return out, paste
