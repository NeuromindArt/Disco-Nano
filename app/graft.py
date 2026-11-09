"""Gives Disco the effect of a prompt, inside its own decoder.

Disco's UNet has no text input. Tiny SD can show what a prompt changes, by running with and without
it and subtracting the features. A linear translator, fitted once on images both models have seen,
turns that difference into Disco's features, and it is added to Disco's decoder at the grids the two
models share.
"""
import os

import torch

import paths

FILE = "prompt_graft.pt"
BINS = 4  # the translator differs with noise level
DISCO_BLOCKS = {16: 8, 32: 5}  # position in output_blocks of the last attention at 1/16 and 1/32 of the image
DISCO_CHANNELS = {16: 512, 32: 1024}


def path():
    return os.path.join(paths.APP_DIR, FILE)


def noise_bin(t):
    """t runs from 0, almost clean, to 999, pure noise."""
    return min(int(t) * BINS // 1000, BINS - 1)


def disco_blocks(model):
    from guided_diffusion.unet import AttentionBlock
    found = {}
    for grid, index in DISCO_BLOCKS.items():
        block = next((layer for layer in model.output_blocks[index] if isinstance(layer, AttentionBlock)), None)
        if block is None or block.channels != DISCO_CHANNELS[grid]:
            raise ValueError("Prompt graft works only with the 512x512 diffusion model.")
        found[grid] = block
    return found


def sd_blocks(unet):
    return {32: unet.up_blocks[0].attentions[-1], 16: unet.up_blocks[1].attentions[-1]}


class Graft:
    def __init__(self, curve, device):
        if not os.path.isfile(path()):
            raise ValueError("Prompt graft needs its translator file.")
        saved = torch.load(path(), map_location="cpu")
        self.maps = {grid: saved["maps"][grid].to(device).float() for grid in saved["maps"]}
        self.curve, self.strength, self.push = curve, 0.0, {}

    def update(self, difference, t):
        """Turns the partner's with-and-without prompt difference into Disco's features."""
        chosen = noise_bin(t)
        self.push = {}
        for grid, features in difference.items():
            moved = features.float().permute(0, 2, 3, 1) @ self.maps[grid][chosen]
            self.push[grid] = moved.permute(0, 3, 1, 2).half()

    def install(self, model):
        """Adds the graft to Disco's blocks. Returns what takes it out again."""
        handles = [block.register_forward_hook(self.hook(grid)) for grid, block in disco_blocks(model).items()]
        return lambda: [handle.remove() for handle in handles]

    def hook(self, grid):
        def add(_, __, output):
            if self.strength and grid in self.push:
                return output + self.strength * self.push[grid].to(output.dtype)
            return output
        return add
