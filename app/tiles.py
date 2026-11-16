"""Tiled generation: one large image built from normal-size renders.

Each tile is an ordinary render, so it needs no more VRAM than the frame size does, and the number
of tiles sets the final size.
"""
import math

import torch
import torch.nn.functional as F
from resize_right import resize

from loaders import device

VIEW_SIDE = 448  # the small copy of the whole picture that CLIP sees, longest side


def grid_of(setting):
    """How many tiles across and down: "3x, 9 tiles" gives 3, "Off" gives 1."""
    return 1 if setting == "Off" else int(setting[0])


def plan(tile_width, tile_height, grid, overlap):
    """The final size, and each tile's top-left corner."""
    step_x, step_y = tile_width - overlap, tile_height - overlap
    final = (tile_width + step_x * (grid - 1), tile_height + step_y * (grid - 1))
    return final, [(col * step_x, row * step_y) for row in range(grid) for col in range(grid)]


def fades(length, step, grid, overlap):
    """How much each column of tiles counts along the picture, or each row. They add up to exactly 1.

    A join that stays put gets sharpened into a straight line by the model, so it moves to a random
    place inside the overlap at every step.
    """
    values = torch.ones(grid, length)
    position = torch.arange(length) + 0.5
    width = max(overlap // 2, 1)
    for i in range(1, grid):
        middle = i * step + width / 2 + torch.rand(()).item() * (overlap - width)
        rising = ((position - middle) / width + 0.5).clamp(0, 1)
        values[i] *= rising
        values[i - 1] *= 1 - rising
    return values


def small_copy(picture):
    """The picture so far, shrunk for CLIP, and how much it was shrunk."""
    height, width = picture.shape[2:]
    scale = VIEW_SIDE / max(width, height)
    size = (max(round(height * scale), 1), max(round(width * scale), 1))
    return F.interpolate(picture, size=size, mode="area"), scale


def placer(view, scale, x, y, width, height, device):
    """A function that puts a tile into the small copy, so CLIP judges it as part of the whole."""
    top, left = round(y * scale), round(x * scale)
    bottom, right = max(round((y + height) * scale), top + 1), max(round((x + width) * scale), left + 1)
    view = view.to(device)

    def place(tile):
        picture = view.clone()
        picture[:, :, top:bottom, left:right] = resize(tile, out_shape=(1, 3, bottom - top, right - left))
        return picture

    return place


def paint(r, start, skip, batch):
    """The render `r`'s tiled picture: every tile through every step together, on one picture held in RAM.

    Each step, every tile predicts its part and the overlaps are averaged before the picture moves on, so
    neighbors settle on the same image instead of being joined at the end. Each tile is an ordinary render
    as far as the GPU is concerned.
    """
    cfg, diffusion, hooks, eraser = r.cfg, r.diffusion, r.hooks, r.eraser
    final, spots, grid, overlap, side_x, side_y = r.final, r.spots, r.grid, r.overlap, r.side_x, r.side_y
    size = (1, 3, final[1], final[0])
    noise = r.start_noise(size, "cpu")
    if noise is None:
        noise = torch.randn(size)
    first = diffusion.num_timesteps - skip - 1
    canvas = noise if skip == 0 and start is None else None
    if canvas is None:
        canvas = diffusion.q_sample(torch.zeros(size) if start is None else start, torch.tensor([first]), noise)
    latest = start

    def paint_tile(source, i, x, y, view):
        """One tile's guided prediction of the picture and of the noise, from `source` at timestep i."""
        box = (..., slice(y, y + side_y), slice(x, x + side_x))
        tile = source[box].to(device)
        r.place = view and cfg["tile_whole_view"] and placer(*view, x, y, side_x, side_y, device)
        r.tile_at = (x, y)
        held = r.pull["to"] if r.pull["set"] else start
        r.init = None if held is None else held[box].to(device)  # Init scale compares against this
        t = torch.tensor([i], device=device)
        with torch.no_grad():
            out = diffusion.p_mean_variance(r.one_pass or r.model, tile, t, clip_denoised=cfg["clip_denoised"], model_kwargs={})
            pred = diffusion.condition_score(r.cond_fn, out, tile, t, model_kwargs={})["pred_xstart"]
            eps = diffusion._predict_eps_from_xstart(tile, t, pred)
            if not i:
                pred = out["pred_xstart"]  # as in a normal render, the last image is the model's own
        return pred.cpu(), eps.cpu()

    def predict_tiles(source, t):
        """Render.predict over the whole picture a tile at a time, the overlaps blended as a step blends them. Tiles
        that no erase touches are left out, as only the erased area is used."""
        across = fades(final[0], side_x - overlap, grid, overlap)
        down = fades(final[1], side_y - overlap, grid, overlap)
        area = eraser.back
        pred_sum, eps_sum = torch.zeros(size), torch.zeros(size)
        for number, (x, y) in enumerate(spots):
            box = (..., slice(y, y + side_y), slice(x, x + side_x))
            if not area[box].any():
                continue
            weight = down[number // grid][y:y + side_y, None] * across[number % grid][x:x + side_x]
            pred, eps = r.predict(source[box], t)
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
        r.t = i
        view = None if latest is None else small_copy(latest)
        pred_sum, eps_sum = torch.zeros(size), torch.zeros(size)
        across = fades(final[0], side_x - overlap, grid, overlap)
        down = fades(final[1], side_y - overlap, grid, overlap)
        for number, (x, y) in enumerate(spots):
            if eraser and not number:
                eraser.shown = latest
            hooks.step(j * len(spots) + number, (first + 1) * len(spots), batch, r.n_batches, r.seed, can_hold=not number,
                       at=1 - i / diffusion.num_timesteps)
            if eraser and not number and latest is not None and eraser.step(latest):
                # A pause holds here, on the last step's picture, which is what the page painted over, so an erase
                # comes before this step paints anything. The canvas is at the last step's noise level, i + 1's.
                start = eraser.cover(start)  # each tile's Init scale compares against its part of this
                latest = eraser.cover(latest)
                r.forget_erased()
                canvas = eraser.restart(canvas, i + 1, predict_tiles)
                view = small_copy(latest)
                shown = r.to_image(latest)
                shown.thumbnail((1024, 1024))
                hooks.preview(shown)  # the erased picture at once, not a step later
            box = (..., slice(y, y + side_y), slice(x, x + side_x))
            weight = down[number // grid][y:y + side_y, None] * across[number % grid][x:x + side_x]
            pred, eps = paint_tile(canvas, i, x, y, view)
            pred_sum[box] += pred * weight
            eps_sum[box] += eps * weight
        r.place = r.tile_at = None
        canvas = ddim(pred_sum, eps_sum, i)
        latest = pred_sum.clamp(-1, 1)
        # A pause holds before the next step's first tile, so the page paints over this step's picture.
        if j % max(cfg["display_rate"] // len(spots), 1) == 0 or not i or r.pausing():
            preview = r.to_image(latest)
            preview.thumbnail((1024, 1024))  # saving a full-size preview takes longer than it's worth
            hooks.preview(preview)
    r.release()
    return latest
