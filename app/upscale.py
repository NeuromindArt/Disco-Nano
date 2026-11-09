"""Upscales one finished image with any ESRGAN, DAT or HAT model that Spandrel can load.

Runs as its own process, so its VRAM is released when it ends. Reports the same @@EVT
lines as renderer.py. Works in overlapping tiles so large models fit in 8GB.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

TILE, OVERLAP = 384, 24


def emit(kind, **data):
    data["type"] = kind
    data["t"] = round(time.time(), 2)
    print("@@EVT " + json.dumps(data), flush=True)


def tiles(size):
    """Start positions covering `size` in TILE-long windows."""
    if size <= TILE:
        return [0]
    stride = TILE - OVERLAP * 2
    starts = list(range(0, size - TILE, stride))
    return starts + [size - TILE]


@torch.inference_mode()
def upscale(model, device, img, half, on_tile):
    _, _, h, w = img.shape
    scale = model.scale
    out = torch.zeros((3, h * scale, w * scale), dtype=torch.float32)
    ys, xs = tiles(h), tiles(w)
    total, done = len(ys) * len(xs), 0
    for y in ys:
        for x in xs:
            tile = img[:, :, y:y + TILE, x:x + TILE].to(device)
            result = model(tile.half() if half else tile).float().clamp(0, 1)[0].cpu()
            # Keep each tile's center, so seams land inside the overlap.
            top = 0 if y == 0 else OVERLAP
            left = 0 if x == 0 else OVERLAP
            bottom = tile.shape[2] if y == ys[-1] else tile.shape[2] - OVERLAP
            right = tile.shape[3] if x == xs[-1] else tile.shape[3] - OVERLAP
            out[:, (y + top) * scale:(y + bottom) * scale, (x + left) * scale:(x + right) * scale] = \
                result[:, top * scale:bottom * scale, left * scale:right * scale]
            done += 1
            on_tile(done, total)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    outcome, detail = "done", {}
    try:
        emit("status", message="Loading upscaler")
        device = torch.device("cuda:0")
        desc = ModelLoader().load_from_file(a.model)
        model = desc.eval().to(device)
        half = desc.supports_half
        if half:
            model.half()
        pixels = np.asarray(Image.open(a.input).convert("RGB")).astype(np.float32) / 255
        img = torch.from_numpy(pixels).permute(2, 0, 1)[None]
        emit("sampling", total_steps=1, n_batches=1)

        def on_tile(done, total):
            emit("step", step=done, total=total, batch=0, n_batches=1)

        out = upscale(model, device, img, half, on_tile)
        array = (out.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        stem = os.path.splitext(a.input)[0]
        path = f"{stem}_x{desc.scale}.png"
        Image.fromarray(array).save(path)
        emit("image", path=path)
    except BaseException as e:
        is_oom = type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()
        outcome = "oom" if is_oom else "error"
        detail = {"error": repr(e)[:600]}
    try:
        detail["peak_vram_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    except Exception:
        pass
    emit("finished", outcome=outcome, **detail)
    sys.exit(0 if outcome == "done" else 1)


if __name__ == "__main__":
    main()
