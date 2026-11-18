"""Experimental changes to the generation loop. Every one is off until its setting is changed.

With all of them off, Render.paint goes through the sampler's own loop in Render.denoise and none of this code runs.
With any on, a picture is painted through denoise() here, the normal ddim loop with room for the ones that work
between steps.
"""
import math

import torch
from resize_right import resize
from torch.nn import functional as F

import settings
from keyframes import schedule_at
from loaders import device


def wanted(cfg):
    return bool(cfg["time_travel"] or cfg["detail_bands"] or cfg["shaped_guidance"] or cfg["gradient_momentum"] or cfg["evidence_map"]
                or cfg["ghost_image"] or cfg["style_image"] or settings.uses_partner(cfg)
                or cfg.get("frame_ghost") and cfg.get("animation_mode", "None") != "None")


def low_pass(tensor, size=16):
    return F.interpolate(F.avg_pool2d(tensor, size), tensor.shape[-2:], mode="bicubic", align_corners=False)


def rms(tensor):
    return tensor.square().mean().sqrt()


class EvidenceMap:
    """Remembers where close-ups score well against the prompt, so a subject can build up in one place."""

    CELL = 16
    WARMUP = 150  # close-ups seen before the map is trusted

    def __init__(self, height, width, strength, device):
        self.strength = strength
        self.total = torch.zeros(height // self.CELL, width // self.CELL, device=device)
        self.count = torch.zeros_like(self.total)
        self.seen = 0

    def add(self, boxes, distances):
        if len(boxes) < 2:
            return
        scores = (distances.mean() - distances) / (distances.std() + 1e-6)  # lower distance is better
        self.total *= 0.99
        self.count *= 0.99
        for (x, y, size), score in zip(boxes, scores.detach()):
            left, top = x // self.CELL, y // self.CELL
            span = max(size // self.CELL, 1)
            self.total[top:top + span, left:left + span] += score
            self.count[top:top + span, left:left + span] += 1
        self.seen += len(boxes)

    @property
    def ready(self):
        return self.seen >= self.WARMUP

    def heat(self):
        mean = (self.total / (self.count + 1)).clamp(min=0)
        return mean / mean.max().clamp(min=1e-6)

    def pick(self, size, side_x, side_y):
        """Half the close-ups are aimed at the map and half stay random, so new areas can still score."""
        if not self.ready or torch.rand([]) < 0.5:
            return None
        chances = self.heat().flatten() + 1e-4
        spot = int(torch.multinomial(chances, 1))
        row, column = spot // self.total.shape[1], spot % self.total.shape[1]
        x = column * self.CELL + self.CELL // 2 - size // 2
        y = row * self.CELL + self.CELL // 2 - size // 2
        return int(min(max(x, 0), side_x - size)), int(min(max(y, 0), side_y - size))

    def weight(self, like):
        heat = self.heat()[None, None]
        return 1 - self.strength * (1 - F.interpolate(heat, like.shape[-2:], mode="bilinear", align_corners=False))


class Lab:
    def __init__(self, cfg, schedule):
        flat = [1.0] * 1000
        self.guidance = schedule(cfg["guidance_curve"]) if cfg["shaped_guidance"] else flat
        self.travel = schedule(cfg["time_travel_curve"]) if cfg["time_travel"] else [0.0] * 1000
        self.detail = schedule(cfg["detail_curve"]) if cfg["detail_bands"] else flat
        self.borrow = schedule(cfg["borrow_curve"]) if cfg["borrow_attention"] else [0.0] * 1000
        self.band = schedule(cfg["painters_curve"]) if cfg["two_painters"] else [0.0] * 1000
        self.band_strength, self.band_color, self.band_path = cfg["painters_strength"], cfg["painters_color"], cfg["painters_path"]
        self.momentum = cfg["gradient_momentum"]
        self.evidence_strength = cfg["evidence_map"]
        self.partner, self.ghost, self.graft = None, None, None
        self.reset(0, 0, None)

    def reset(self, height, width, device):
        self.average = None
        self.evidence = EvidenceMap(height, width, self.evidence_strength, device) if self.evidence_strength and device else None

    def repeats(self, at):
        return int(self.travel[at])

    def shape_push(self, grad, at):
        """Changes CLIP's push on the image before it is sent back through the UNet."""
        if self.momentum:
            self.average = grad if self.average is None else self.momentum * self.average + (1 - self.momentum) * grad
            grad = self.average
        if self.detail[at] != 1:
            broad = low_pass(grad)
            grad = broad + self.detail[at] * (grad - broad)
        if self.evidence is not None and self.evidence.ready:
            grad = grad * self.evidence.weight(grad)
        return grad

    def share(self, sketch, estimate, band):
        """How far the shared estimate moves, when the partner owns the shapes broader than band pixels."""
        change = low_pass(sketch.to(estimate.dtype) - estimate, band)
        return self.band_strength * (change if self.band_color else change.mean(1, keepdim=True))


class Ghost:
    """Guides a render with an image that never reaches it as pixels.

    Pushes are scaled to the size of CLIP's push, so their strengths mean the same at any setting.
    """

    CROPS = 8

    def __init__(self, cfg, schedule, ghost, style, clip_models, normalize, depth=None):
        self.normalize = normalize
        self.depth, self.depth_strength = depth, cfg["depth_target"]
        self.relief, self.relief_of = None, None
        self.curve = schedule(cfg["ghost_curve"])
        self.structure, self.crops, self.lowfreq = cfg["ghost_structure"], cfg["ghost_crops"], cfg["ghost_lowfreq"]
        self.style_strength = cfg["style_strength"]
        self.ghost = ghost  # 0 to 1
        # The robust checker is an open_clip model, whose layers are laid out differently from OpenAI's CLIP.
        clip_models = [m for m in clip_models if not getattr(m, "robust", False)]
        vits = [m for m in clip_models if hasattr(m.visual, "transformer")]
        nets = [m for m in clip_models if hasattr(m.visual, "layer1")]
        self.vit = min(vits, key=lambda m: m.visual.conv1.kernel_size[0], default=None)  # the finest one
        self.net = nets[0] if nets else None
        needs_vit = ghost is not None and (self.structure or self.crops)
        if needs_vit and self.vit is None:
            raise ValueError("Ghost guidance needs a ViT CLIP model.")
        if style is not None and self.net is None:
            raise ValueError("Style image needs a ResNet checker.")
        with torch.no_grad():
            self.shapes = self.relations(ghost) if needs_vit else None
            self.style = self.textures(style) if style is not None else None

    def clip_input(self, image, size):
        return self.normalize(resize(image.float(), out_shape=[image.shape[0], 3, size, size]))

    def relations(self, image):
        """How every patch of the image relates to every other, which holds its shapes and not its look."""
        seen = []
        hook = self.vit.visual.transformer.resblocks[-2].register_forward_hook(lambda m, i, out: seen.append(out))
        self.vit.encode_image(self.clip_input(image, self.vit.visual.input_resolution))
        hook.remove()
        patches = F.normalize(seen[0][1:].permute(1, 0, 2).float(), dim=-1)
        return patches @ patches.transpose(1, 2)

    def textures(self, image):
        """Gram matrices of a ResNet checker's first layers, the classic measure of texture and palette."""
        seen = []
        visual = self.net.visual
        hooks = [layer.register_forward_hook(lambda m, i, out: seen.append(out)) for layer in (visual.layer1, visual.layer2, visual.layer3)]
        self.net.encode_image(self.clip_input(image, visual.input_resolution))
        for hook in hooks:
            hook.remove()
        grams = []
        for features in seen:
            b, c, h, w = features.shape
            flat = features.float().reshape(b, c, h * w)
            grams.append(flat @ flat.transpose(1, 2) / (h * w))
        return grams

    def areas(self, image):
        side_y, side_x = image.shape[2:4]
        painted, ghost = [], []
        size_in = self.vit.visual.input_resolution
        for _ in range(self.CROPS):
            size = int(min(side_x, side_y) * (0.3 + 0.4 * float(torch.rand([]))))
            x = int(torch.randint(0, side_x - size + 1, ()))
            y = int(torch.randint(0, side_y - size + 1, ()))
            painted.append(self.clip_input(image[:, :, y:y + size, x:x + size], size_in))
            ghost.append(self.clip_input(self.ghost[:, :, y:y + size, x:x + size], size_in))
        with torch.no_grad():
            target = F.normalize(self.vit.encode_image(torch.cat(ghost)).float(), dim=-1)
        embeds = F.normalize(self.vit.encode_image(torch.cat(painted)).float(), dim=-1)
        return (1 - (embeds * target).sum(-1)).mean()

    def push(self, x_in, clip_grad, at):
        image = x_in.add(1).div(2)
        size = rms(clip_grad)
        total = clip_grad

        def add(loss, strength):
            grad = torch.autograd.grad(loss, x_in, retain_graph=True)[0]  # later terms go back through the same image
            return total + strength * grad * size / rms(grad).clamp(min=1e-20)

        weight = self.curve[at]
        if self.shapes is not None and self.structure and weight:
            total = add((self.relations(image) - self.shapes).square().mean(), self.structure * weight)
        if self.ghost is not None and self.crops and weight:
            total = add(self.areas(image), self.crops * weight)
        if self.style is not None and self.style_strength:
            mine = self.textures(image)
            total = add(sum((a - b).square().mean() for a, b in zip(mine, self.style)), self.style_strength)
        if self.depth and self.ghost is not None and weight:
            if self.relief_of is not self.ghost:  # a new sketch means a new target
                with torch.no_grad():
                    self.relief, self.relief_of = self.depth.relief(self.ghost), self.ghost
            total = add((self.depth.relief(image) - self.relief).abs().mean(), self.depth_strength * weight)
        return total

    def pull(self, x, estimate, scale, at):
        """Moves the render's broad light and dark toward the ghost's, and leaves color and detail alone."""
        strength = self.lowfreq * self.curve[at]
        if self.ghost is None or not strength:
            return x
        target = self.ghost.mul(2).sub(1).to(estimate.dtype)
        change = low_pass(target - estimate).mean(1, keepdim=True)
        return x + strength * scale * change


def denoise(r, start, width, height, skip, batch, stop=None, show=None):
    """The render `r`'s ddim loop for one picture, with room for the experiments that work between steps: the
    Partner's sketch and borrowed attention, Prompt graft, a ghost, Detail bands and Correct steps. `stop` and `show`
    work as in Render.paint."""
    cfg, diffusion, hooks, experiment, eraser, steer, judge = r.cfg, r.diffusion, r.hooks, r.experiment, r.eraser, r.steer, r.judge
    show = show or r.to_image
    r.init = r.pull["to"] if r.pull["set"] else start
    before, r.ahead = r.ahead, 0
    experiment.reset(height, width, device)
    helper = experiment.partner
    ghost = experiment.ghost
    needed = cfg["two_painters"] or cfg["live_ghost"] == "Live" or cfg["borrow_attention"] or cfg["prompt_graft"]
    grafting = experiment.graft
    sketching = helper and cfg["live_ghost"] == "Sketch first" and r.given["ghost"] is None
    if sketching and r.given["sketch"] is not None:  # a clip keeps its first sketch, or the layout would jump every frame
        ghost.ghost, sketching = r.given["sketch"], False
    if helper and (needed or sketching):
        helper.to(device)
        helper.restart(r.seed + batch)
        if sketching:
            hooks.status("Painting the sketch")
            drawn = helper.paint(cfg["sketch_steps"])
            ghost.ghost = drawn.add(1).div(2).float()
            if r.animating:
                r.given["sketch"] = ghost.ghost
            if cfg["partner_keep_sketch"]:
                r.save_sketch(drawn)
        if not needed:  # only the sketch was wanted, so the render gets the VRAM back
            helper.to("cpu")
            torch.cuda.empty_cache()
    if r.given["ghost"] is not None:
        ghost.ghost = r.given["ghost"]
    first = diffusion.num_timesteps - skip - 1
    x = r.start_noise((1, 3, height, width), device)
    if x is None:
        x = torch.randn(1, 3, height, width, device=device)
    if skip or start is not None:
        base = torch.zeros_like(x) if start is None else start
        x = diffusion.q_sample(base, torch.tensor([first], device=device), x)
    image = torch.zeros_like(x)
    for j, i in enumerate(range(first, -1, -1)):
        r.t = i
        if eraser:
            eraser.shown = image  # the last step's picture, which a pause here shows
        hooks.step(j + before, first + 1 + before, batch, r.n_batches, r.seed, at=1 - i / diffusion.num_timesteps)
        t = torch.tensor([i], device=device)
        at = schedule_at(i, diffusion.num_timesteps)
        band = int(experiment.band[at])
        if helper and needed:
            helper.strength = experiment.borrow[at]
            live = cfg["live_ghost"] == "Live" and ghost.curve[at]
            if grafting:
                grafting.strength = cfg["prompt_graft"] * grafting.curve[at]
            if band or (helper.strength or live or grafting and grafting.strength) and j % cfg["partner_every"] == 0:
                helper.refresh(image, float(diffusion.alphas_cumprod[i]))
                if live:
                    ghost.ghost = helper.sketch().add(1).div(2).float()
                if grafting:
                    grafting.update(helper.difference, helper.nearest(float(diffusion.alphas_cumprod[i])))
        alpha, alpha_prev = diffusion.alphas_cumprod[i], diffusion.alphas_cumprod_prev[i]
        repeats = experiment.repeats(at)
        for repeat in range(repeats + 1):
            with torch.no_grad():
                out = diffusion.ddim_sample(r.one_pass, x, t, clip_denoised=cfg["clip_denoised"], cond_fn=r.cond_fn,
                                            model_kwargs={}, eta=cfg["eta"])
            x = out["sample"]
            if repeat < repeats:  # back up one step, so the model can absorb the push
                ratio = alpha / alpha_prev
                x = math.sqrt(ratio) * x + math.sqrt(1 - ratio) * torch.randn_like(x)
        image = out["pred_xstart"].detach()
        if ghost:
            x = ghost.pull(x, image, math.sqrt(alpha_prev), at)
        x = r.hold_layout(x, image, i)
        if eraser and eraser.step(image):
            r.forget_erased()
            x = eraser.restart(x, i, r.predict)
        if steer:
            x = steer.pull(x, image, i, diffusion)
        if band:
            moved = experiment.share(helper.sketch(), image, band)
            image = image + moved
            x = x + math.sqrt(alpha_prev) * moved
            if experiment.band_path == "Blend":  # the noise estimate changes with the mix, so the step stays small
                sigma = cfg["eta"] * math.sqrt((1 - alpha_prev) / (1 - alpha)) * math.sqrt(1 - alpha / alpha_prev)
                x = x - math.sqrt(max(1 - alpha_prev - sigma ** 2, 0)) * math.sqrt(alpha / (1 - alpha)) * moved
        if judge and cfg["correct_steps"] and judge.correct_curve[at] and i > 0:
            # Langevin steps at the new noise level, so the picture settles there before the next step. With
            # Two painters they aim at both models' combined prediction, so the path samples both at once.
            r.t = i - 1
            t_next = torch.tensor([i - 1], device=device)
            for _ in range(int(cfg["correct_steps"])):
                with torch.no_grad():
                    mine = diffusion.ddim_sample(r.one_pass, x, t_next, clip_denoised=cfg["clip_denoised"],
                                                 cond_fn=r.cond_fn if cfg["correct_clip"] else None, model_kwargs={}, eta=0)["pred_xstart"]
                    if helper and band:
                        theirs = helper.denoise(mine, float(alpha_prev))
                        mine = mine + experiment.share(theirs, mine, band)
                    x = judge.corrector(x, mine, float(alpha_prev), float(cfg["correct_snr"]))
            image = mine.detach()
        if j % cfg["display_rate"] == 0 or not i or j + 1 == stop or r.pausing():
            hooks.preview(show(image))
        if j + 1 == stop:
            break
    if helper and needed and helper.z0 is not None and cfg["partner_keep_sketch"]:
        r.save_sketch(helper.sketch())
    r.release()
    return image
