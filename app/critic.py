"""Experiments that change the critic, CLIP, rather than bringing Stable Diffusion's structure in.

CLIP is as happy with subject texture spread over the canvas as with one whole subject, which is the
root of Disco's scattered subjects. Each experiment here is off until its setting changes. They are
in the Experiments card, and their settings are in settings.py with the other experiments.

- Prompt routing: the subject is scored on whole-image views and the style on close-ups.
- Coherence critic: each cut is also scored with its tiles shuffled, and only the arrangement counts.
- Robust checker: an adversarially fine-tuned CLIP (FARE, Schlarmann et al., MIT), whose push draws
  shapes, leads early.
- One subject: CLIP's patch map of the subject (the MaskCLIP trick, Zhou et al.) is pushed toward one place.
- Correctors: Langevin steps at each noise level, with the step size rule of Song et al.'s predictor-corrector
  sampler, alone or making Two painters sample both models at once.
"""
import torch
from torch.nn import functional as F

from guidance import spherical_dist_loss

GRIDS = {"All": (16, 32, 64), "Coarse": (32, 64), "Coarsest": (64,)}

SUBJECT, STYLE, BOTH = 0, 1, 2


def wanted(cfg):
    return bool(cfg["route_prompts"] != "Off" or cfg["layout_critic"] or cfg["robust_critic"] != "Off" or cfg["one_subject"]
                or cfg["correct_steps"])


# ---- routed prompts ----
def split_lines(lines, mode, parse):
    """Splits each prompt into its subject and its style. Returns the new lines and each one's role."""
    out, roles = [], []
    for index, line in enumerate(lines):
        text, weight = parse(line)
        if mode == "First line":
            out.append(line)
            roles.append(SUBJECT if index == 0 else STYLE)
            continue
        subject, _, style = text.partition(",")
        out.append(f"{subject.strip()}:{weight}")
        roles.append(SUBJECT)
        if style.strip():
            out.append(f"{style.strip()}:{weight}")
            roles.append(STYLE)
    return out, roles


def subject_text(cfg, lines, parse):
    words = [w.strip() for w in str(cfg["subject_words"]).split(";") if w.strip()]
    return words or [parse(lines[0])[0].partition(",")[0].strip()]


# ---- the coherence critic ----
def shuffle_tiles(batch, generator):
    """Cuts every image into a grid of 2 to 4 tiles a side, at a random offset, and moves every tile."""
    n, c, h, w = batch.shape
    out = []
    for i in range(n):
        grid = int(torch.randint(2, 5, (), generator=generator))
        tile = h // grid
        shift_y, shift_x = (int(v) for v in torch.randint(0, tile, (2,), generator=generator))
        image = torch.roll(batch[i:i + 1], (shift_y, shift_x), (2, 3))[:, :, :tile * grid, :tile * grid]
        tiles = image.unfold(2, tile, tile).unfold(3, tile, tile).permute(0, 2, 3, 1, 4, 5).reshape(1, grid * grid, c, tile, tile)
        while True:
            order = torch.randperm(grid * grid, generator=generator)
            if (order != torch.arange(grid * grid)).all():
                break
        tiles = tiles[:, order].reshape(1, grid, grid, c, tile, tile).permute(0, 3, 1, 4, 2, 5).reshape(1, c, tile * grid, tile * grid)
        out.append(F.interpolate(tiles, size=(h, w), mode="bilinear", align_corners=False))
    return torch.cat(out)


# ---- the robust critic ----
def load_robust(name, device, status=print):
    import open_clip

    import models
    file, network = models.ROBUST_CHOICES[name]
    path = models.ensure(file, status)
    status(f"Loading {name}")
    model = open_clip.create_model(network, pretrained=path, device="cpu").eval().requires_grad_(False).half().to(device)
    size = model.visual.image_size
    model.visual.input_resolution = size[0] if isinstance(size, (tuple, list)) else size
    encode = model.encode_image
    dtype = next(model.visual.parameters()).dtype
    model.encode_image = lambda image: encode(image.to(dtype))
    model.robust = True
    return model


# ---- one subject ----
def dense_map(model, image):
    """MaskCLIP: the last block's value path, without attention, gives patch tokens that line up with text."""
    visual = model.visual
    x = visual.conv1(image.to(visual.conv1.weight.dtype))
    grid = x.shape[-1]
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], 1) + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x).permute(1, 0, 2)
    blocks = visual.transformer.resblocks
    for block in blocks[:-1]:
        x = block(x)
    last = blocks[-1]
    y = last.ln_1(x)
    width = y.shape[-1]
    v = F.linear(y, last.attn.in_proj_weight[2 * width:], last.attn.in_proj_bias[2 * width:])
    v = last.attn.out_proj(v)
    tokens = visual.ln_post(v.permute(1, 0, 2)[:, 1:]) @ visual.proj
    return F.normalize(tokens.float(), dim=-1), grid


def blob_outside(similarity):
    """1 outside the largest region where the subject scores high, 0 inside it and just around it."""
    values = similarity.detach().float()
    high = (values > values.mean() + 0.5 * values.std()).cpu().tolist()
    rows, cols = len(high), len(high[0])
    seen, best = set(), []
    for r in range(rows):
        for c in range(cols):
            if not high[r][c] or (r, c) in seen:
                continue
            region, todo = [], [(r, c)]
            seen.add((r, c))
            while todo:
                y, x = todo.pop()
                region.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < rows and 0 <= nx < cols and high[ny][nx] and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        todo.append((ny, nx))
            if len(region) > len(best):
                best = region
    if not best:
        return None
    keep = torch.zeros(1, 1, rows, cols, device=similarity.device)
    for y, x in best:
        keep[0, 0, y, x] = 1
    keep = F.max_pool2d(keep, 3, stride=1, padding=1)[0, 0]  # and one cell around it
    return 1 - keep


class Critic:
    def __init__(self, cfg, schedule, seed):
        self.cfg = cfg
        self.route = cfg["route_prompts"]
        self.route_curve = schedule(cfg["route_curve"])
        self.layout, self.margin = float(cfg["layout_critic"]), float(cfg["layout_margin"])
        self.layout_curve = schedule(cfg["layout_curve"])
        self.robust_mode = cfg["robust_mode"]
        self.robust_curve = schedule(cfg["robust_curve"]) if cfg["robust_critic"] != "Off" else None
        self.subject = float(cfg["one_subject"])
        self.subject_curve = schedule(cfg["one_subject_curve"])
        self.subject_model, self.subject_embeds = None, None
        self.correct_curve = schedule(cfg["correct_curve"])
        self.generator = torch.Generator().manual_seed(seed)  # its own numbers, so shuffling never changes the render's

    # how much each checker counts at this step
    def model_weight(self, model, at):
        if self.robust_curve is None:
            return 1.0
        share = self.robust_curve[at]
        if getattr(model, "robust", False):
            return share
        return 1.0 - share if self.robust_mode == "Hand over" else 1.0

    def prompt_weights(self, stat, total, overview, at):
        """[cuts, prompts] weights when prompts are routed by cut scale, or None for the usual weights."""
        roles = stat.get("roles")
        if self.route == "Off" or roles is None:
            return None
        weights, roles = stat["weights"], roles.to(stat["weights"].device)
        is_subject, is_style = (roles != STYLE).float(), (roles != SUBJECT).float()
        whole = weights * is_subject
        close = weights * (is_style if self.route == "Split" else torch.ones_like(is_style))
        c = self.route_curve[at]
        whole, close = c * whole + (1 - c) * weights, c * close + (1 - c) * weights
        rows = torch.stack([whole] * overview + [close] * (total - overview))
        return rows / rows.abs().sum(1, keepdim=True).clamp(min=1e-6)

    # the coherence critic
    def layout_on(self, at):
        return self.layout and self.layout_curve[at] > 0

    def layout_grad(self, stat, batch, dists, overview, n, x_in, chunk, guidance, at, row_weights):
        """The push that makes shuffled cuts score worse than whole ones. Leaves batch's graph in place."""
        count = overview * n if self.cfg["layout_cuts"] == "Whole-image" else batch.shape[0]
        if not count:
            return 0
        shuffled = shuffle_tiles(batch[:count], self.generator)
        leaf = shuffled.detach().requires_grad_()
        whole = dists.detach().view(-1, n, dists.shape[-1])[:count // n]
        if row_weights is None:
            weighted = (whole * stat["weights"]).sum(-1)
        else:
            weighted = (whole * row_weights[:count // n, None, :]).sum(-1)
        cut_grad = 0
        for i in range(0, count, chunk):
            embeds = stat["model"].encode_image(leaf[i:i + chunk]).float()
            d = spherical_dist_loss(embeds.unsqueeze(1), stat["embeds"].unsqueeze(0)).view(-1, n, len(stat["weights"]))
            rows = slice(i // n, i // n + d.shape[0])
            w = stat["weights"] if row_weights is None else row_weights[rows, None, :]
            mixed = (d * w).sum(-1)
            gap = mixed - weighted[rows]
            loss = F.relu(self.margin - gap).sum() / max(count // n, 1)
            cut_grad = cut_grad + torch.autograd.grad(loss * guidance * self.layout * self.layout_curve[at], leaf)[0]
        return torch.autograd.grad(shuffled, x_in, cut_grad, retain_graph=True)[0]

    # one subject
    def prepare_subjects(self, stats, words, tokenize, device):
        """Picks the finest ViT and encodes the subject words, before the text encoders leave the GPU."""
        if not self.subject:
            return
        vits = [s["model"] for s in stats if hasattr(s["model"].visual, "transformer") and not getattr(s["model"], "robust", False)]
        if not vits:
            raise ValueError("One subject needs a ViT checker.")
        self.subject_model = min(vits, key=lambda m: m.visual.conv1.kernel_size[0])
        with torch.no_grad():
            embeds = self.subject_model.encode_text(tokenize(words, truncate=True).to(device)).float()
        self.subject_embeds = F.normalize(embeds, dim=-1)

    def subject_push(self, x_in, clip_grad, at, normalize):
        weight = self.subject * self.subject_curve[at]
        if not weight or self.subject_model is None:
            return clip_grad
        size = self.subject_model.visual.input_resolution
        image = normalize(F.interpolate(x_in.add(1).div(2), size=(size, size), mode="bilinear", align_corners=False))
        tokens, grid = dense_map(self.subject_model, image)
        maps = (tokens @ self.subject_embeds.T)[0].T.reshape(-1, grid, grid)  # [subjects, grid, grid]
        loss, parts = 0, 0
        for k in range(maps.shape[0]):
            outside = blob_outside(maps[k])
            if outside is not None:
                level = maps[k].detach().mean()
                loss = loss + (F.relu(maps[k] - level) * outside).mean()
                parts += 1
        for a in range(maps.shape[0]):
            for b in range(a + 1, maps.shape[0]):
                la, lb = maps[a].detach().mean(), maps[b].detach().mean()
                loss = loss + (F.relu(maps[a] - la) * F.relu(maps[b] - lb)).mean() * 10
                parts += 1
        if not parts or not torch.is_tensor(loss) or not loss.requires_grad:
            return clip_grad
        grad = torch.autograd.grad(loss, x_in, retain_graph=True)[0]
        rms = lambda t: t.square().mean().sqrt()  # noqa: E731
        return clip_grad + weight * grad * rms(clip_grad) / rms(grad).clamp(min=1e-20)

    # correctors
    def corrector(self, x, x0, alpha_bar, snr):
        """One Langevin step toward the combined prediction x0, at noise level alpha_bar."""
        score = (alpha_bar ** 0.5 * x0 - x) / (1 - alpha_bar)
        noise = torch.randn_like(x)
        step = 2 * (snr * noise.norm() / score.norm().clamp(min=1e-12)) ** 2
        return x + step * score + (2 * step) ** 0.5 * noise
