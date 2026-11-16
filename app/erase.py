"""Fix: erases what was painted over, and clones one part of the picture over another, during a render or on a
start image before one.

During a render, LaMa fills the marked area of the picture so far, and the area carries on from that fill, so the
steps left repaint it in Disco's own way and the seam disappears. The page writes the mask, then erase.json, into the
job's folder as the pause ends, and the render takes it up before the next step. A later pause can erase again, when
the prompt grows something new there.

The area's noise decides whether the erased thing stays gone, and how much fresh noise it gets depends on how far
through the render the erase is. Up to FRESH_FROM it is all fresh: the step's own noise is still strong then and
holds the old thing's shape, and the next steps would rebuild it in the same place. From FRESH_TO there is none, only
the step's own: by then that noise is too weak to rebuild anything, and fresh noise can't all be painted out in the
steps left, so it turns to speckle in the area and grain across the rest of the tile. Between the two, the share of
fresh noise falls in a straight line. Tested at 768x512 with Tiles, erasing a lighthouse: with only the step's own
noise it came back at 60%, 70% and 75%, and not at 80%. All fresh noise at 70% and half at 75% kept it gone with no
grain, while all fresh noise at 75% spread grain across the tile.

With fresh noise, the area first goes back BACK of the schedule and comes down again in BACK_STEPS quick steps of
the model alone, before the render carries on. Partway through a render, the steps only build on the broad shapes
an area already holds, and LaMa fills smoothly, so a fill carried straight on is painted as haze. Back at more
noise, the model draws broad shapes of its own, and the area keeps as much detail as what is around it. Tested on
the same render: a fill carried straight on lost its detail within a few steps, while a detailed piece of the
picture put in its place kept it, as did the fill after going back 40%, but not 25%. A late erase has too few steps
left after coming down to add detail, so it doesn't go back.

With Tiles, the picture is the whole tiled image, held in RAM, so LaMa fills a crop around what was painted over,
made smaller if it is very large. Each erase keeps the picture before it and LaMa's fill in the job's folder, as
erase-N-before.jpg and erase-N-fill.jpg, to see what it did.

Inpaint has a model of your own repaint an area, in inpaint.py, after any erase and before any clone in the same
pause. Like a clone, it is as detailed as the picture, so it gets fresh noise but doesn't go back to be drawn again.

Clone copies part of the picture over another, as painted with a source offset, after any erase in the same pause.
Each source the page picked is a layer: a soft mask and how far away it copies from. The copy's colors are matched to
where it lands, by the difference along its edge spread smoothly across it, so a piece from a lighter part of the
sky doesn't show as a patch. A clone gets fresh noise as an erase does, so what it covers doesn't come back, but it
doesn't go back to be drawn again: it is already as detailed as the picture, and tested at 85%, a detailed piece of
the picture put over a lighthouse kept its detail to the end and blended in, where LaMa's fill was painted as haze.
Late in a render, cloning is the way to fix something and keep detail.

What the render is held to forgets the thing too, through cover(): the start image Init scale compares against,
Small first's picture and the Controls' picture. Otherwise they would keep pulling it back.

Run as a script, it fixes an image file, for a start image before a render:
    erase.py --image start.png [--mask mask.png] [--clones clones.json] [--inpaint inpaint.json] --out fixed.png [--device cpu]
clones.json is a list of {"mask": path, "dx": share of the width, "dy": share of the height}.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

import models

# How far through the schedule, 0 to 1, an erase's fresh noise starts to fall, and where none is left
FRESH_FROM, FRESH_TO = 0.7, 0.8
# How far back through the schedule an erased area goes to draw its broad shapes again, and in how many steps
BACK, BACK_STEPS = 0.4, 10


def masks(mask, width, height, device):
    """A mask image as the hard mask LaMa fills, grown a little so it has room around what it removes, and a soft
    copy whose edge blends. Both are 1x1xHxW tensors of 0 to 1."""
    mask = mask.convert("L").resize((width, height), Image.BILINEAR)
    grow = max(3, min(width, height) // 64)
    hard = mask.point(lambda v: 255 if v > 127 else 0).filter(ImageFilter.MaxFilter(grow * 2 + 1))
    soft = hard.filter(ImageFilter.GaussianBlur(grow))
    as_tensor = lambda m: torch.from_numpy(np.asarray(m, dtype=np.float32) / 255)[None, None].to(device)
    return as_tensor(hard), as_tensor(soft)


def lama(picture, mask, device, status):
    """LaMa's fill of the masked area, for a picture of 0 to 1. It loads for this and leaves the GPU straight after."""
    path = models.ensure(models.LAMA, status)
    model = torch.jit.load(path, map_location=device).eval()
    h, w = picture.shape[-2:]
    pad = lambda t: F.pad(t, (0, -w % 8, 0, -h % 8), mode="reflect")
    try:
        with torch.no_grad():
            out = model(pad(picture), pad(mask))[..., :h, :w].float().clamp(0, 1)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return picture * (1 - mask) + out * mask


def fill(picture, mask, device, status, longest=1536):
    """The picture, 0 to 1, with the masked area filled, from a crop around it: the mask's box and as much again
    around it, for LaMa to see what surrounds it. A crop longer than `longest` is filled smaller and enlarged back."""
    h, w = picture.shape[-2:]
    rows, cols = torch.nonzero(mask[0, 0].sum(1)).flatten(), torch.nonzero(mask[0, 0].sum(0)).flatten()
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    grow_y, grow_x = max(bottom - top, 64), max(right - left, 64)
    top, bottom = max(top - grow_y, 0), min(bottom + grow_y, h)
    left, right = max(left - grow_x, 0), min(right + grow_x, w)
    crop = (..., slice(top, bottom), slice(left, right))
    part, hole = picture[crop].to(device), mask[crop].to(device)
    ch, cw = part.shape[-2:]
    scale = min(longest / max(ch, cw), 1.0)
    if scale < 1:
        small = (max(round(ch * scale), 8), max(round(cw * scale), 8))
        filled = lama(F.interpolate(part, small, mode="area"), (F.interpolate(hole, small, mode="area") > 0).float(), device, status)
        filled = F.interpolate(filled, (ch, cw), mode="bicubic", align_corners=False).clamp(0, 1)
    else:
        filled = lama(part, hole, device, status)
    out = picture.clone()
    out[crop] = (part * (1 - hole) + filled * hole).to(picture.device)
    return out


def blur(x, s):
    """A Gaussian blur of width s, across and then down, with the edges mirrored."""
    r = max(int(3 * s), 1)
    k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * s * s))
    k = (k / k.sum())
    c = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    x = F.pad(x, (0, 0, r, r), mode="replicate")
    return F.conv2d(x, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def clone(picture, layers):
    """The picture, 0 to 1, with each clone layer copied over it, and the soft mask of where they went. A layer is
    (mask, dx, dy): a 1x1xHxW mask of 0 to 1, and how far right and down the source is, in pixels. Every layer copies
    from the picture as it was, as the page showed it."""
    h, w = picture.shape[-2:]
    out, where = picture.clone(), torch.zeros_like(picture[:, :1])
    for mask, dx, dy in layers:
        if not mask.any():
            continue
        # The source, moved onto the brush: each pixel takes the one dx, dy away, the nearest edge one past the frame.
        ys = (torch.arange(h, device=picture.device) + dy).clamp(0, h - 1)
        xs = (torch.arange(w, device=picture.device) + dx).clamp(0, w - 1)
        copy = picture[..., ys, :][..., xs]
        # The color difference along a thin ring around the mask, spread smoothly over it, about as far as the
        # stroke is wide: worked out at a quarter of the size, which is plenty for something this smooth.
        small = (max(h // 4, 1), max(w // 4, 1))
        inside = F.interpolate(mask, small, mode="area")
        touched = (inside > 0.02).float()
        ring = F.max_pool2d(touched, 5, 1, 2) - F.max_pool2d(touched, 3, 1, 1)
        width = float(((inside > 0.5).float().sum() / max(float(ring.sum()), 1.0)) * 2) + 2  # area / edge, in small pixels
        gap = F.interpolate(picture - copy, small, mode="area") * ring
        match = blur(gap, width) / blur(ring, width).clamp(min=1e-4)
        match = F.interpolate(match, (h, w), mode="bilinear", align_corners=False)
        out = out * (1 - mask) + (copy + match).clamp(0, 1) * mask
        where = torch.maximum(where, mask)
    return out, where


def read_clones(clones, width, height, device):
    """The page's clone layers, as clone() takes them, for a picture of this size."""
    layers = []
    for layer in clones or []:
        mask = Image.open(layer["mask"]).convert("L").resize((width, height), Image.BILINEAR)
        mask = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255)[None, None].to(device)
        layers.append((mask, round(float(layer["dx"]) * width), round(float(layer["dy"]) * height)))
    return layers


def keep(picture, path):
    """Saves a -1 to 1 picture as a JPEG, smaller than 2048 on its long side."""
    image = Image.fromarray((picture[0].permute(1, 2, 0).add(1).div(2).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    image.thumbnail((2048, 2048))
    image.save(path, quality=90)


class Erase:
    def __init__(self, job_dir, device, diffusion, status, inpainter=None):
        self.job_dir, self.path = job_dir, os.path.join(job_dir, "erase.json")
        self.device, self.diffusion, self.status = device, diffusion, status
        # inpainter(picture, hard, soft, asked): the picture, 0 to 1, with the area repainted as the page asked
        self.inpainter = inpainter
        self.read = None
        self.count = 0
        self.last = None  # the last fix's picture and soft mask, for cover()
        self.back = None  # the last fix's erased area, which goes back to be drawn again, as a soft mask
        self.cloned = None  # the last fix's cloned and inpainted area, which carries on as it is, as a soft mask
        self.fresh = None  # the share of fresh noise the page chose for the last erase, or None to go by the schedule
        self.erased = False  # an erase has just happened, and what the render is held to hasn't been covered yet

    def request(self):
        """A new erase.json, once, as its erase mask or None, its clone layers, its inpaint or None, and the share of
        fresh noise the page chose, or None; or None with nothing new."""
        try:
            changed = os.path.getmtime(self.path)
            if changed == self.read:
                return None
            with open(self.path, encoding="utf-8") as f:
                asked = json.load(f)
            asked = asked.get("mask"), list(asked.get("clones") or []), asked.get("inpaint"), asked.get("fresh")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return None  # none, or one being written
        self.read = changed
        return asked

    def step(self, estimate):
        """Erases, inpaints and clones what the page asked for on the picture so far, `estimate`, and returns True, or False
        with nothing new. The caller covers what the render is held to, then calls restart()."""
        asked = self.request()
        if asked is None:
            return False
        mask, clones, inpainting, self.fresh = asked
        estimate = estimate.detach().float()
        h, w = estimate.shape[-2:]
        picture = estimate.add(1).div(2).clamp(0, 1)
        back = torch.zeros_like(picture[:, :1])
        where = cloned = back
        if mask:
            hard, back = masks(Image.open(mask), w, h, estimate.device)
            if hard.any():
                self.status("Erasing the marked area")
                picture = fill(picture, hard, self.device, self.status)
                where = back
        if inpainting:
            hard, soft = masks(Image.open(inpainting["mask"]), w, h, estimate.device)
            if hard.any():
                start = fill(picture, hard, self.device, self.status) if inpainting["mode"] == "remove" else picture
                try:
                    picture = self.inpainter(start, hard, soft, inpainting).to(picture.device, picture.dtype)
                    cloned = soft
                    back = back * (1 - soft)  # painted over an erase: the inpaint is what goes there
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:  # the render carries on: a Remove as an erase, the others as they were
                    reason = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
                    if inpainting["mode"] == "remove":
                        self.status(f"Inpainting failed, so the area was erased instead: {reason}")
                        picture, back = start, torch.maximum(back, soft)
                    else:
                        self.status(f"Inpainting failed, so the area was left as it was: {reason}")
                        soft = torch.zeros_like(soft)
                where = torch.maximum(where, soft)
        layers = read_clones(clones, w, h, estimate.device)
        if layers:
            self.status("Cloning")
            copied, layered = clone(estimate.add(1).div(2).clamp(0, 1), layers)
            picture = picture * (1 - layered) + copied * layered
            back = back * (1 - layered)  # cloned over an erase: the copy is what goes there, and needs no drawing again
            where = torch.maximum(where, layered)
            cloned = torch.maximum(cloned, layered)
        if not where.any():
            return False
        filled = picture.mul(2).sub(1)
        keep(estimate, os.path.join(self.job_dir, f"erase-{self.count}-before.jpg"))
        keep(filled, os.path.join(self.job_dir, f"erase-{self.count}-fill.jpg"))
        self.count += 1
        self.last, self.back, self.cloned, self.erased, self.estimate = (filled, where), back, cloned, True, estimate
        return True

    def restart(self, x, i, predict=None):
        """The sample x, which the step at timestep index i has just made, with the fixed areas carrying on from the
        fill and the clones, their noise partly or all fresh by how late the fix is; see the top of this file.
        `predict(x, t)` gives the model's own, unguided picture and noise for a sample at timestep t, for the erased
        area to go back and draw its broad shapes again."""
        filled, soft = self.last[0].to(x.device, torch.float32), self.last[1].to(x.device, torch.float32)
        estimate = self.estimate.to(x.device)
        alphas = self.diffusion.alphas_cumprod
        alpha = float(self.diffusion.alphas_cumprod_prev[i])
        at = 1 - i / self.diffusion.num_timesteps
        share = min(max((FRESH_TO - at) / (FRESH_TO - FRESH_FROM), 0.0), 1.0) if self.fresh is None else self.fresh
        own = (x.float() - math.sqrt(alpha) * estimate) / math.sqrt(1 - alpha)  # the step's own noise
        picture = estimate * (1 - soft) + filled * soft
        noise, fresh = own, None
        if share:
            # Its own generator, so the render's own random draws, and so the rest of the picture, stay as they were.
            fresh = torch.randn(filled.shape, generator=torch.Generator().manual_seed(self.count)).to(x.device)
            # Mixed by square roots, so the noise is as strong as the step expects everywhere, across the soft edge too.
            # Weaker noise there would be cleaned as if it were full strength, and take the picture's detail with it.
            mix = soft * share
            noise = (1 - mix).sqrt() * own + mix.sqrt() * fresh
        now = i - 1  # the timestep the next step paints at
        far = min(now + round(BACK * self.diffusion.num_timesteps), self.diffusion.num_timesteps - 1)
        back, cloned = self.back.to(x.device, torch.float32), self.cloned.to(x.device, torch.float32)
        if not share or predict is None or now < 0 or far <= now or not back.any():
            return (math.sqrt(alpha) * picture + math.sqrt(1 - alpha) * noise).to(x.dtype)
        # LaMa fills smoothly, and the steps left only build on the broad shapes an area holds, so a smooth fill is
        # painted as haze. Back at more noise, the model draws broad shapes of its own in the area, fitting what is
        # around it, and the area comes back down to this step. The rest of the picture is shown to it at each noise
        # level from its own noise, and is left exactly as it was, and a clone from its copy and its noise.
        self.status("Erasing the marked area: drawing it again")
        around = estimate * (1 - cloned) + filled * cloned
        around_noise = own if fresh is None else (1 - cloned * share).sqrt() * own + (cloned * share).sqrt() * fresh
        times = sorted({round(far - (far - now) * k / BACK_STEPS) for k in range(BACK_STEPS + 1)}, reverse=True)
        level = float(alphas[far])
        sample = math.sqrt(level) * picture + math.sqrt(1 - level) * noise
        for t, t_next in zip(times, times[1:]):
            pred, eps = predict(sample.to(x.dtype), t)
            level = float(alphas[t_next])
            drawn = math.sqrt(level) * pred.float() + math.sqrt(1 - level) * eps.float()
            sample = (math.sqrt(level) * around + math.sqrt(1 - level) * around_noise) * (1 - back) + drawn * back
        return sample.to(x.dtype)

    def cover(self, picture, unit=False):
        """A picture the render is held to, with the last erase's fill over the same place, at the picture's own
        size. `unit` is for a picture of 0 to 1, rather than -1 to 1."""
        if picture is None or self.last is None:
            return picture
        filled, soft = self.last
        if filled.shape[-2:] != picture.shape[-2:]:
            filled = F.interpolate(filled, picture.shape[-2:], mode="bilinear", align_corners=False)
            soft = F.interpolate(soft, picture.shape[-2:], mode="bilinear", align_corners=False)
        if unit:
            filled = filled.add(1).div(2)
        filled, soft = filled.to(picture.device, picture.dtype), soft.to(picture.device, picture.dtype)
        return picture * (1 - soft) + filled * soft


def erase_image(image_path, mask_path, clones, out_path, device, status=print, inpainting=None):
    """Erases from an image file what the mask marks, if there is one, inpaints as `inpainting` asks, as the page sends
    it with its mask as a file, clones the layers over it, and saves the result as a PNG."""
    image = Image.open(image_path).convert("RGB")
    picture = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]
    original = picture
    if mask_path:
        hard, _ = masks(Image.open(mask_path), image.width, image.height, "cpu")
        if hard.any():
            picture = fill(picture, hard, device, status)
    if inpainting:
        hard, soft = masks(Image.open(inpainting["mask"]), image.width, image.height, "cpu")
        if hard.any():
            import random
            import handoff
            import inpaint
            if inpainting["mode"] == "remove":
                picture = fill(picture, hard, device, status)
            unit = lambda m: Image.fromarray((m[0, 0].numpy() * 255).round().astype(np.uint8))
            model = handoff.load(inpainting["file"], status, device)
            painted = inpaint.paint(model, inpaint.to_pil(picture), unit(hard), unit(soft), inpainting["prompt"],
                                    inpainting["strength"], inpainting["steps"], inpainting["guidance"],
                                    random.randint(0, 2**32 - 1), status)
            picture = inpaint.from_pil(painted, picture)
    layers = read_clones(clones, image.width, image.height, "cpu")
    if layers:
        copied, cloned = clone(original, layers)
        picture = picture * (1 - cloned) + copied * cloned
    Image.fromarray((picture[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)).save(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask")
    parser.add_argument("--clones", help="a JSON file of clone layers")
    parser.add_argument("--inpaint", help="a JSON file of what to inpaint, as the server writes it")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    clones = []
    if args.clones:
        with open(args.clones, encoding="utf-8") as f:
            clones = json.load(f)
    inpainting = None
    if args.inpaint:
        with open(args.inpaint, encoding="utf-8") as f:
            inpainting = json.load(f)
    erase_image(args.image, args.mask, clones, args.out, torch.device(args.device), lambda message: print(message, flush=True),
                inpainting)
