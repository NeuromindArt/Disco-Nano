"""The prompts a render follows: text lines with weights, and folders of images, as each CLIP model sees them.

Derived from Disco Diffusion, which is MIT licensed. See CREDITS.md and LICENSES/disco-diffusion.txt for its authors.
"""
import os

import clip
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
from torch.nn import functional as F

import critic
from guidance import normalize
from imaging import upright
from loaders import device


def parse_prompt(prompt):
    text, _, weight = prompt.rpartition(":")
    try:
        return (text, float(weight)) if text else (prompt, 1.0)
    except ValueError:
        return prompt, 1.0


IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
MAX_PROMPT_IMAGES = 200


def prompt_images(folder):
    """The images in an image prompt folder, at most MAX_PROMPT_IMAGES of them, in name order."""
    if not folder:
        return []
    if not os.path.isdir(folder):
        raise ValueError("The image prompt folder can't be found. Choose it again under Guide images.")
    found = sorted(os.path.join(folder, n) for n in os.listdir(folder) if n.lower().endswith(IMAGE_TYPES))
    if not found:
        raise ValueError("The image prompt folder has no PNG, JPEG, WebP or BMP images.")
    return found[:MAX_PROMPT_IMAGES]


@torch.no_grad()
def image_prompt_embed(model, files):
    """What a folder of images has in common, as one CLIP model sees it: the average of every image's
    embedding, each read whole and mirrored. Like a learned embedding, but nothing is trained."""
    side = model.visual.input_resolution
    total = None
    for path in files:
        image = ImageOps.fit(upright(path), (side, side), Image.LANCZOS)
        batch = TF.to_tensor(image).unsqueeze(0)
        batch = normalize(torch.cat([batch, batch.flip(3)])).to(device)
        embeds = F.normalize(model.encode_image(batch).float(), dim=-1).sum(0, keepdim=True)
        total = embeds if total is None else total + embeds
    return F.normalize(total, dim=-1)


def encode_prompts(clip_models, prompts, cfg, image_embeds=None, roles=None):
    """roles, when prompts are routed by cut scale, says whether each line is subject or style."""
    stats = []
    for index, model in enumerate(clip_models):
        embeds, weights, kinds = [], [], []
        if image_embeds:
            embeds.append(image_embeds[index])
            weights.append(float(cfg["image_prompt_weight"]))
            kinds.append(critic.BOTH)
        for number, prompt in enumerate(prompts):
            text, weight = parse_prompt(prompt)
            embed = model.encode_text(clip.tokenize(text, truncate=True).to(device)).float()
            kind = roles[number] if roles else critic.BOTH
            if cfg["fuzzy_prompt"]:
                for _ in range(25):
                    embeds.append((embed + torch.randn(embed.shape, device=device) * cfg["rand_mag"]).clamp(0, 1))
                    weights.append(weight)
                    kinds.append(kind)
            else:
                embeds.append(embed)
                weights.append(weight)
                kinds.append(kind)
        weights = torch.tensor(weights, device=device)
        if weights.sum().abs() < 1e-3:
            raise ValueError("The prompt weights must not add up to 0.")
        stats.append({"model": model, "embeds": torch.cat(embeds), "weights": weights / weights.sum().abs(),
                      **({"roles": torch.tensor(kinds)} if roles else {})})
    return stats
