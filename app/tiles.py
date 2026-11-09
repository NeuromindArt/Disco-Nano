"""Tiled generation: one large image built from normal-size renders.

Each tile is an ordinary render, so it needs no more VRAM than the frame size does, and the number
of tiles sets the final size.
"""
import torch
import torch.nn.functional as F
from resize_right import resize

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
