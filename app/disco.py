"""Disco Diffusion's render path, for still images and animation frames: one render or clip, from its settings to
its saved images.

Derived from Disco Diffusion, which is MIT licensed. See CREDITS.md and LICENSES/disco-diffusion.txt for its authors.
Their animation, video and notebook code is not included; animation here comes from animate.py, which is Nano's own.
The cutouts and losses are in guidance.py, the secondary model in secondary.py, loading and keeping the models in
loaders.py and the prompts in prompts.py. renderer.py runs this in its own process.

A render is a Render: what it was asked for, the models it loaded and where it has got to. Its steps are here. The
parts that only some renders take have their own modules and are handed the Render: a clip's frames in
animate.film, the experiments' loop in lab.denoise and the tiles in tiles.paint.
"""
import gc
import json
import math
import os
import random
import re

import clip
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.nn import functional as F

import animate
import control
import critic
import erase
import depth
import graft
import imaging
import keyframes
import lab
import memory
import models
import partner
import paths
import rules
import tiles
from guidance import MakeCutoutsDango, forget_lpips_sizes, lpips_distance, normalize, range_loss, spherical_dist_loss, tv_loss
from imaging import size_start_image, upright
from keyframes import POINTS, schedule, schedule_at
from loaders import OnePassUNet, after_render, device, kept, load_clip_models, load_lpips, load_secondary, load_unet, put_away
from prompts import encode_prompts, image_prompt_embed, parse_prompt, prompt_images
from secondary import alpha_sigma_to_t
from settings import DEFAULTS, handoff_parts, small_first_on, small_frame, uses_partner


def next_run_number(folder, batch_name):
    number = 0
    while os.path.exists(os.path.join(folder, f"{batch_name}({number})_settings.txt")):
        number += 1
    return number


def set_seeds(seed):
    seed %= 2**32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def render(job_settings, prompts, hooks, offload=False, init_fit="crop", output_dir=paths.OUTPUT):
    """One render or clip. Whatever happens, the models it used go back to RAM for the next render."""
    try:
        Render(job_settings, prompts, hooks, offload, init_fit, output_dir).run()
    finally:
        put_away()


class Render:
    """One render or clip: its settings, the models it loaded, and where it has got to."""

    def __init__(self, job_settings, prompts, hooks, offload, init_fit, output_dir):
        """Reads the settings, loads the models and prepares everything the steps follow: the prompts and regions,
        the experiments, the controls and Fix. Nothing is painted yet."""
        self.hooks, self.offload, self.init_fit = hooks, offload, init_fit
        cfg = self.cfg = {**DEFAULTS, **job_settings}
        forget_lpips_sizes()
        style = models.cut_style(cfg["diffusion_model"])
        if style:
            cfg.update({k: v for k, v in models.CUT_SCHEDULES[style].items() if k not in job_settings})
        side_x, side_y = self.side_x, self.side_y = [(side // 64) * 64 for side in cfg.get("width_height", [768, 512])]
        self.secondary = cfg["use_secondary_model"]
        self.grid = tiles.grid_of(cfg["tile_scale"])
        self.overlap = min(int(cfg["tile_overlap"]), min(side_x, side_y) // 2)
        self.final, self.spots = tiles.plan(side_x, side_y, self.grid, self.overlap)
        clash = rules.conflicts(cfg)
        if clash:
            raise ValueError(clash[0])
        # Small first: the first small_steps of the path are painted at `small`, in the frame's shape.
        self.small = small_frame(side_x, side_y, int(cfg["small_side"])) if self.grid == 1 and small_first_on(cfg) else None
        self.small_steps = min(round(cfg["steps"] * float(cfg["small_first"]) / 100), cfg["steps"] - 1) if self.small else 0
        trying = lab.wanted(cfg) or critic.wanted(cfg)
        self.animating = cfg["animation_mode"] != "None"
        mirrored = cfg["use_horizontal_symmetry"] or cfg["use_vertical_symmetry"]
        if trying and (self.grid > 1 or self.secondary or mirrored or cfg["diffusion_sampling_mode"] != "ddim"):
            raise ValueError("Experimental settings need the ddim sampler, with no tiles, symmetry or secondary model.")
        if self.grid > 1:
            cfg["diffusion_sampling_mode"] = "ddim"  # plms can't hold finished pixels still
        self.batch_name = cfg.get("batch_name", "Disco")
        self.folder = os.path.join(output_dir, self.batch_name)
        os.makedirs(self.folder, exist_ok=True)

        self.seed = seed = random.randint(0, 2**32 - 1) if cfg["set_seed"] == "random_seed" else int(cfg["set_seed"])
        set_seeds(seed)
        self.sets = sets = keyframes.prompt_sets(prompts) if self.animating else [(0, prompts)]
        self.handing = None
        self.hand_start, self.hand_finish = handoff_parts(cfg)
        if self.hand_start or self.hand_finish:
            import handoff  # loaded before Disco, so its text encoders can use the GPU and leave
            path = cfg["handoff_file"]
            self.handing = kept(("handoff", path, os.path.getmtime(path)), lambda: handoff.load(path, hooks.status, device))
            self.handing.prepare(cfg, sets, hooks.status, getattr(hooks, "stopped", lambda: False), hooks.preview)
        torch.backends.cudnn.benchmark = False  # same speed here, but its cached workspaces cost VRAM

        self.overview = schedule(cfg["cut_overview"])
        self.innercut = schedule(cfg["cut_innercut"])
        self.ic_pow = schedule(cfg["cut_ic_pow"])
        self.icgray_p = schedule(cfg["cut_icgray_p"])
        self.views_from = 0  # where on the POINTS-point schedule the picture being painted began, for fit_views

        clip_models = self.clip_models = load_clip_models(cfg, hooks.status)
        self.model, self.diffusion = load_unet(cfg, hooks)
        self.secondary_model = load_secondary(hooks) if self.secondary else None
        pictures = prompt_images(cfg["image_prompts"]) if float(cfg["image_prompt_weight"]) else []
        if pictures:
            hooks.status(f"Reading {len(pictures)} prompt images")
        image_embeds = [image_prompt_embed(m, pictures) for m in clip_models] if pictures else None
        judge = self.judge = critic.Critic(cfg, schedule, seed) if critic.wanted(cfg) else None
        if judge and judge.route != "Off":
            routed = [(at, *critic.split_lines(lines, cfg["route_split"], parse_prompt)) for at, lines in sets]
            self.all_stats = [(at, encode_prompts(clip_models, lines, cfg, image_embeds, roles)) for at, lines, roles in routed]
        else:
            self.all_stats = [(at, encode_prompts(clip_models, lines, cfg, image_embeds)) for at, lines in sets]
        stats = self.stats = self.all_stats[0][1]
        if judge:
            judge.prepare_subjects(stats, critic.subject_text(cfg, sets[0][1], parse_prompt), clip.tokenize, device)
        self.regions = {"rev": None, "boxes": [], "strength": 0.0, "embeds": {}, "dirty": False}
        self.take_regions()
        memory.free_text_encoders(clip_models)

        self.start_image = cfg.get("init_image") or None
        self.init_scale, self.skip_steps = cfg["init_scale"], cfg["skip_steps"]
        self.init, self.lpips_model, self.source = None, None, None  # source: the start image, sized to the frame
        if self.start_image:
            self.source = size_start_image(upright(self.start_image), *self.final, init_fit)
            if self.init_scale:
                self.lpips_model = load_lpips()

        drawing = cfg["live_ghost"] != "Off"
        if cfg["prompt_graft"] and cfg["partner_model"] != "Tiny SD":
            raise ValueError("Prompt graft needs Tiny SD as the partner model.")
        self.frame_ghost = self.animating and cfg["frame_ghost"]
        if cfg["depth_target"] and not (drawing or cfg["ghost_image"] or self.frame_ghost):
            raise ValueError("Depth target follows a ghost. Set Live ghost, choose a Ghost image, or turn on Last frame as ghost in a clip.")
        experiment = self.experiment = lab.Lab(cfg, schedule) if trying else None

        if uses_partner(cfg):
            text = ", ".join(parse_prompt(p)[0] for p in sets[0][1] if parse_prompt(p)[1] > 0)
            kind, file = cfg["partner_model"], cfg["partner_file"] if cfg["partner_model"] != "Tiny SD" else ""
            weights = kept(("partner", kind, file, os.path.getmtime(file) if os.path.isfile(file) else 0),
                           lambda: partner.Weights(kind, file, hooks.status))
            experiment.partner = partner.Partner(weights, text, side_y, side_x, cfg["partner_follow"], device)
            after_render(experiment.partner.release)
            scales = critic.GRIDS.get(cfg["borrow_grids"], critic.GRIDS["All"])
            experiment.partner.lengths = {(side_y // s) * (side_x // s) for s in scales}
            partner.install(experiment.partner)
            after_render(partner.uninstall)
        if cfg["prompt_graft"]:
            experiment.graft = graft.Graft(schedule(cfg["graft_curve"]), device)
            after_render(experiment.graft.install(self.model))
            experiment.partner.watch(graft.sd_blocks(experiment.partner.unet))
        if cfg["ghost_image"] or cfg["style_image"] or drawing or self.frame_ghost:
            def sized(path):
                image = size_start_image(upright(path), side_x, side_y, "crop")
                return TF.to_tensor(image).unsqueeze(0).to(device)

            experiment.ghost = lab.Ghost(cfg, schedule, sized(cfg["ghost_image"]) if cfg["ghost_image"] else None,
                                         sized(cfg["style_image"]) if cfg["style_image"] else None, [s["model"] for s in stats], normalize,
                                         self.depth_model() if cfg["depth_target"] else None)

        self.steer = None  # the controls: edges, depth and pose held to a control picture
        self.fixed_control = None  # the Control image, when one is set; otherwise the start image or each frame's video frame
        if control.wanted(cfg):
            self.steer = control.Control(cfg, schedule, device, hooks.status, self.depth_model, kept)
            if cfg.get("control_image") and not self.animating:  # a clip follows each frame, never one fixed picture
                self.fixed_control = TF.to_tensor(size_start_image(upright(cfg["control_image"]), side_x, side_y, init_fit)).unsqueeze(0)

        self.t = None  # the timestep being painted
        # in animation, a ghost made from the last frame, the clip's one sketch, and a frame's own start noise
        self.given = {"ghost": None, "sketch": None, "noise": None}
        self.place = None  # while tiling, puts the tile into a small copy of the whole picture
        self.tile_at = None  # while tiling, where the tile is in the whole picture, so regions measure its close-ups there
        self.pull = {"set": False, "to": None}  # after a handoff, Init scale still compares against the start before it
        self.ahead = 0  # steps Small first already painted, so the full-size pass counts on from them
        self.hold_to = None  # after Small first, the small picture the full size holds its broad shapes to
        self.hold_curve = schedule(cfg["small_hold_curve"]) if self.small and float(cfg["small_hold"] or 0) else None
        self.hold_size = max(8, min(side_x, side_y) // 24)  # shapes broader than this are held: 32 pixels at 1280x768

        # Fix: what was painted over during a pause, erased, inpainted or cloned, and repainted by the steps left.
        self.eraser = erase.Erase(hooks.job_dir, device, self.diffusion, hooks.status, self.inpainter, hooks.tried, kept) \
            if getattr(hooks, "job_dir", None) else None
        if self.eraser:
            hooks.while_paused = lambda: self.eraser.try_paint(hooks.pausing)  # inpaint tries, painted while the render waits
        self.one_pass = OnePassUNet(self.model) if cfg["diffusion_sampling_mode"] == "ddim" and not self.secondary else None

        self.run_number = next_run_number(self.folder, self.batch_name)
        self.resume = cfg.get("resume_clip") if self.animating else None  # the last frame of a clip to carry on
        if self.resume:
            match = re.fullmatch(r"(.*)\((\d+)\)_0\.png", os.path.basename(self.resume))
            if not match:
                raise ValueError("Only an animation's own image can be continued.")
            self.folder, self.batch_name, self.run_number = os.path.dirname(self.resume), match.group(1), int(match.group(2))
        self.n_batches = cfg["n_batches"]
        # Tile denoise is only for Render first, where the tiles repaint the picture its first pass made.
        self.tile_skip = min(round(cfg["steps"] * (1 - cfg["tile_denoise"])), cfg["steps"] - 1)

    # ---- running ----
    def run(self):
        """Paints the stills, or the clip."""
        self.hooks.status("Starting the first step")
        if self.animating:
            animate.film(self)
        else:
            self.stills()

    def stills(self):
        cfg, seed, hooks = self.cfg, self.seed, self.hooks
        side_x, side_y = self.side_x, self.side_y
        for batch in range(self.n_batches):
            gc.collect()
            torch.cuda.empty_cache()
            set_seeds(seed + batch)  # each image can be made again on its own
            start = self.start_tensor()
            if self.steer:
                self.steer.set_source(self.fixed_control if self.fixed_control is not None else start.add(1).div(2) if start is not None else None)
            handoff_picture = cfg["handoff_keep"] and self.named(f"_{batch}_handoff.png")
            if self.grid == 1:
                self.fit_views(self.skip_steps)
                start, skip = self.hand_over(start, self.skip_steps, batch, (side_x, side_y), seed + batch, handoff_picture)
                set_seeds(seed + batch)
                start, skip = self.small_start(start, skip, batch)
                image = self.to_image(self.paint(start, side_x, side_y, skip, batch))
            else:
                if not self.start_image and cfg["tile_start"] == "Render first":
                    self.fit_views(self.skip_steps)
                    start, skip = self.hand_over(None, self.skip_steps, batch, (side_x, side_y), seed + batch, handoff_picture)
                    set_seeds(seed + batch)
                    first = self.to_image(self.denoise(start, side_x, side_y, skip, batch))
                    start, skip = TF.to_tensor(first.resize(self.final, Image.LANCZOS)).unsqueeze(0).mul(2).sub(1), self.tile_skip
                    self.fit_views(self.tile_skip)
                else:
                    self.fit_views(self.skip_steps)
                    start, skip = self.hand_over(start, self.skip_steps, batch, self.final, seed + batch, handoff_picture)
                    set_seeds(seed + batch)
                image = self.to_image(tiles.paint(self, None if start is None else start.cpu(), skip, batch))
            disco_own = cfg["handoff_keep"] and self.named(f"_{batch}_disco.png")
            image = self.finish(image, batch, seed + batch, disco_own)
            hooks.image(self.save(image, batch))

    def start_tensor(self):
        """The start image as a -1 to 1 tensor on the GPU, or None without one."""
        return TF.to_tensor(self.source).unsqueeze(0).mul(2).sub(1).to(device) if self.start_image else None

    # ---- the views ----
    # The views schedules lay out the scene in their first 40% and paint detail after. Shapes share spreads that
    # first part over a share of the steps, and Fit views spreads both over only the steps a picture paints, so one
    # that starts partway, from a start image or a video frame, still lays out shapes before its detail.
    def fit_views(self, skip, frame=False):
        """`frame` is a clip's frame, which is fitted only with Start images and clip frames."""
        mode = self.cfg["views_fit"]
        if isinstance(mode, bool):  # settings saved while it was a switch
            mode = "Start images and clip frames" if mode else "Off"
        fit = mode == "Start images and clip frames" or (mode == "Start images" and not frame)
        self.views_from = min(round(POINTS * skip / self.diffusion.num_timesteps), POINTS - 1) if fit else 0

    def view_at(self, at):
        begin, shapes, pivot = self.views_from, min(max(float(self.cfg["views_shapes"]) / 100, 0), 1), 400
        done = min(max((at - begin) / (1000 - begin), 0), 1)
        if done < shapes:
            point = done / shapes * pivot
        else:
            point = pivot + (done - shapes) / max(1 - shapes, 1e-6) * (1000 - pivot)
        return min(int(point), POINTS - 1)

    # ---- regions ----
    def take_regions(self):
        """Regional prompts: boxes with prompts of their own. Encodes them at the start, and again when a pause
        changed them. Returns True when it changed them."""
        hooks, regions, clip_models = self.hooks, self.regions, self.clip_models
        rev = getattr(hooks, "regions_rev", 0)
        if rev == regions["rev"]:
            return False
        given = hooks.regions if hasattr(hooks, "regions") else self.cfg.get("regions")
        boxes = [b for b in (given or {}).get("boxes", []) if str(b.get("text", "")).strip()]
        moving = regions["rev"] is not None and boxes  # the text encoders have left the GPU by now
        regions.update(rev=rev, boxes=boxes, strength=float((given or {}).get("strength", 0.75)), embeds={})
        if regions["rev"] or given:
            self.cfg["regions"] = given
            regions["dirty"] = True
        if moving:
            memory.text_encoders_back(clip_models, device)
        with torch.no_grad():
            for model in clip_models:
                tokens = clip.tokenize([parse_prompt(b["text"])[0] for b in boxes], truncate=True).to(device) if boxes else None
                if tokens is not None:
                    regions["embeds"][model] = model.encode_text(tokens).float()
        if moving:
            memory.free_text_encoders(clip_models)
        return True

    def region_rows(self, main, boxes, overview, total, height, width):
        """[cuts, prompts + regions] weights. A close-up is judged by each region's prompt in proportion to how much
        of it the region covers, times Region strength, and by the main prompts for the rest. A whole-image view
        is judged the same way, by how much of the frame each region covers.

        With Tiles, the regions lie over the whole tiled picture, and a tile's close-ups are measured where the tile
        is in it. A tile's whole-image view is the whole picture when Whole picture view shows the tile in it, and
        otherwise only the tile."""
        left, top = self.tile_at or (0, 0)
        full_w, full_h = self.final if self.tile_at else (width, height)
        rects = torch.tensor([[b["x"] * full_w, b["y"] * full_h, (b["x"] + b["w"]) * full_w, (b["y"] + b["h"]) * full_h]
                              for b in self.regions["boxes"]], device=device)
        seen = [0, 0, full_w, full_h] if self.tile_at is None or self.place else [left, top, left + width, top + height]
        spots = torch.tensor([seen] * overview + [[left + x, top + y, left + x + size, top + y + size] for x, y, size in boxes],
                             device=device, dtype=torch.float32)
        across = (torch.minimum(spots[:, None, 2], rects[None, :, 2]) - torch.maximum(spots[:, None, 0], rects[None, :, 0])).clamp(min=0)
        down = (torch.minimum(spots[:, None, 3], rects[None, :, 3]) - torch.maximum(spots[:, None, 1], rects[None, :, 1])).clamp(min=0)
        cover = across * down / ((spots[:, 2] - spots[:, 0]) * (spots[:, 3] - spots[:, 1]))[:, None]
        cover = cover / cover.sum(1, keepdim=True).clamp(min=1)  # overlapping regions share the cut
        share = self.regions["strength"] * cover
        main = main.expand(total, -1) if main.dim() == 1 else main
        return torch.cat([main * (1 - share.sum(1, keepdim=True)), share], 1)

    # ---- models it shares ----
    def depth_model(self):
        """Depth Anything: one copy for Depth target, Depth control, Face shape and the 3D camera, on the GPU for the
        render and kept in RAM between renders, as Disco's other models are."""
        return kept(("depth",), lambda: depth.Depth(device, self.hooks.status)).to(device)

    def resting(self):
        """The render's models that wait in RAM while a handoff or inpaint model paints, if the two won't fit."""
        return [self.model, self.secondary_model, self.lpips_model, *(m.visual for m in self.clip_models),
                self.experiment.partner if self.experiment else None]

    def inpainter(self, picture, hard, asked):
        """Fix's Inpaint: the picture, 0 to 1, with the area repainted by the Inpaint model the page chose, and the
        soft mask it went back in through. The model is kept for the rest of the render, shared with a handoff that
        uses the same file, so later inpaints and tries don't load it again, and let go when the render ends unless
        the handoff uses it. Disco waits in RAM while it paints if the two won't fit on the GPU together."""
        import handoff
        import inpaint
        hooks = self.hooks
        path = asked["file"]
        painter = kept(("handoff", path, os.path.getmtime(path)), lambda: handoff.load(path, hooks.status, device), brief=True)
        image, size = imaging.to_pil(picture), (picture.shape[-1], picture.shape[-2])
        resting = self.resting()
        self.release()
        with handoff.Parked(painter, resting, device, *painter.inpaint_size(*size), painted=True):
            painted, paste = inpaint.paint(painter, image, imaging.mask_to_pil(hard), asked.get("prompt", ""), asked["strength"],
                                           asked["steps"], asked["guidance"], asked.get("seed", self.seed + self.eraser.count), hooks.status,
                                           getattr(hooks, "stopped", lambda: False),  # a new try paints anew
                                           asked.get("context", inpaint.CONTEXT), asked.get("grow", inpaint.GROW), asked.get("blend", inpaint.BLEND))
        return imaging.from_pil(painted, picture), imaging.mask_from_pil(paste, picture.device)

    # ---- a step ----
    def start_noise(self, shape, where):
        """The frame's own start noise, when it has one of this shape, or None for a fresh draw."""
        noise = self.given["noise"]
        if noise is None or tuple(noise.shape) != tuple(shape):
            return None
        return noise.to(where)

    def predict(self, x, t):
        """The model's own picture and noise for x at timestep t, without guidance, on x's device: for an erase."""
        diffusion = self.diffusion
        tt = torch.tensor([t], device=device)
        with torch.no_grad():
            out = diffusion.p_mean_variance(self.one_pass or self.model, x.to(device), tt, clip_denoised=self.cfg["clip_denoised"], model_kwargs={})
            eps = diffusion._predict_eps_from_xstart(x.to(device), tt, out["pred_xstart"])
        return out["pred_xstart"].to(x.device), eps.to(x.device)

    def forget_erased(self):
        """After an erase, what the render is held to forgets the thing too: the start image Init scale compares
        against, Small first's picture, the Controls' picture and a ghost. Otherwise they would keep pulling it back."""
        eraser, steer = self.eraser, self.steer
        eraser.erased = False
        self.init = eraser.cover(self.init)
        self.pull["to"] = eraser.cover(self.pull["to"])
        self.hold_to = eraser.cover(self.hold_to)
        if steer and steer.source is not None:
            steer.set_source(eraser.cover(steer.source, unit=True))
        ghost = self.experiment.ghost if self.experiment else None
        if ghost and ghost.ghost is not None:  # a Ghost image or a sketch made first; Live ghost redraws from the render
            ghost.ghost = eraser.cover(ghost.ghost, unit=True)
            if ghost.shapes is not None:
                with torch.no_grad():
                    ghost.shapes = ghost.relations(ghost.ghost)

    def hold_layout(self, x, estimate, i):
        """Hold the small layout: after the step at index i, moves the render's broad shapes and colors toward the
        small picture, as a ghost's light and dark is moved, and leaves the detail free."""
        if self.hold_to is None or self.hold_curve is None or i == 0:
            return x
        diffusion = self.diffusion
        at = schedule_at(i, diffusion.num_timesteps)
        strength = float(self.cfg["small_hold"]) * self.hold_curve[at]
        if not strength:
            return x
        scale = math.sqrt(float(diffusion.alphas_cumprod_prev[i]))
        return x + (strength * scale * lab.low_pass(self.hold_to - estimate.detach().float(), self.hold_size)).to(x.dtype)

    def cond_fn(self, x, t, y=None):
        """CLIP guidance for one step: how x should move toward the prompts, the regions, the controls and the
        experiments' targets, and away from losses such as Init scale's."""
        cfg, diffusion, one_pass, judge, experiment, steer = self.cfg, self.diffusion, self.one_pass, self.judge, self.experiment, self.steer
        cur_t = self.t
        self.take_regions()  # before the step's graph is built, as it may bring the text encoders back for a moment
        with torch.enable_grad():
            x = one_pass.leaf if one_pass else x.detach().requires_grad_()
            n = x.shape[0]
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            if self.secondary:
                alpha = torch.tensor(diffusion.sqrt_alphas_cumprod[cur_t], device=device, dtype=torch.float32)
                sigma = torch.tensor(fac, device=device, dtype=torch.float32)
                pred = self.secondary_model(x, alpha_sigma_to_t(alpha, sigma)[None].repeat([n])).pred
            else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                pred = diffusion.p_mean_variance(one_pass or self.model, x, my_t, clip_denoised=False,
                                                 model_kwargs={"y": y})["pred_xstart"]
            x_in = pred * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            at = 1000 - (int(t.item()) + 1)
            view = self.view_at(at)  # the views schedules' point, which Fit views and Shapes share can move
            guidance = cfg["clip_guidance_scale"] * experiment.guidance[at] if experiment else cfg["clip_guidance_scale"]
            overview, innercut = int(self.overview[view]), int(self.innercut[view])
            for stat in self.stats:
                share = judge.model_weight(stat["model"], at) if judge else 1.0
                if not share:
                    continue  # a checker with no say at this step costs no time
                own = self.regions["embeds"][stat["model"]] if self.regions["boxes"] else None
                # with regions, their prompts join the main ones, and each cut weighs them by how much it covers
                full = stat if own is None else {**stat, "embeds": torch.cat([stat["embeds"], own]),
                                                 "weights": F.pad(stat["weights"], (0, len(own)))}
                for _ in range(cfg["cutn_batches"]):
                    faces = steer.face_crops(x_in, at) if steer else []  # Face detail: close-ups on each face
                    cuts = MakeCutoutsDango(stat["model"].visual.input_resolution, overview, innercut,
                                            self.ic_pow[view], self.icgray_p[view], cfg["skip_augs"], cfg["exact_repeat"],
                                            experiment.evidence if experiment else None, faces)
                    whole = self.place(x_in).add(1).div(2) if self.place and overview else None
                    batch = normalize(cuts(x_in.add(1).div(2), whole))
                    total = overview + innercut + len(faces)
                    chunk = memory.clip_chunk(stat["model"]) * n
                    rows = judge.prompt_weights(stat, total, overview, at) if judge else None
                    if own is not None:
                        rows = self.region_rows(stat["weights"] if rows is None else rows, cuts.boxes, overview, total,
                                                x_in.shape[2], x_in.shape[3])
                    weights = full["weights"] if rows is None else rows[:, None, :]  # per cut when prompts are routed or regions set
                    layout = judge is not None and judge.layout_on(at)
                    if chunk >= batch.shape[0]:
                        image_embeds = stat["model"].encode_image(batch).float()
                        dists = spherical_dist_loss(image_embeds.unsqueeze(1), full["embeds"].unsqueeze(0))
                        dists = dists.view([total, n, -1])
                        if layout:
                            x_in_grad += share * judge.layout_grad(full, batch, dists, overview, n, x_in, chunk, guidance,
                                                                   at, rows) / cfg["cutn_batches"]
                        losses = dists.mul(weights).sum(2).mean(0)
                        x_in_grad += torch.autograd.grad(losses.sum() * guidance * share, x_in)[0] / cfg["cutn_batches"]
                    else:  # a few cuts at a time through a big CLIP, then one pass back through the cutting
                        leaf, cut_grad, parts = batch.detach().requires_grad_(), 0, []
                        for i in range(0, leaf.shape[0], chunk):
                            image_embeds = stat["model"].encode_image(leaf[i:i + chunk]).float()
                            part = spherical_dist_loss(image_embeds.unsqueeze(1), full["embeds"].unsqueeze(0)).view([-1, n, len(full["weights"])])
                            w = weights if rows is None else weights[i // n:i // n + part.shape[0]]
                            loss = part.mul(w).sum(2).sum(0).sum() / total
                            cut_grad = cut_grad + torch.autograd.grad(loss * guidance * share, leaf)[0]
                            parts.append(part.detach())
                        dists = torch.cat(parts)
                        if layout:
                            x_in_grad += share * judge.layout_grad(full, batch, dists, overview, n, x_in, chunk, guidance,
                                                                   at, rows) / cfg["cutn_batches"]
                        x_in_grad += torch.autograd.grad(batch, x_in, cut_grad)[0] / cfg["cutn_batches"]
                    if cuts.evidence:
                        cuts.evidence.add(cuts.boxes, dists.mul(full["weights"]).sum(2)[overview:, 0])
            if experiment:
                x_in_grad = experiment.shape_push(x_in_grad, at)
                if experiment.ghost:
                    x_in_grad = experiment.ghost.push(x_in, x_in_grad, at)
            if steer:
                x_in_grad = steer.push(x_in, x_in_grad, at)
            if judge:
                x_in_grad = judge.subject_push(x_in, x_in_grad, at, normalize)
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            # Disco Diffusion measures range on pred, which x_in is made from, so its gradient with respect to x_in
            # is always zero and Color range does nothing. Measured on x_in, it pulls colors back into range.
            loss = (tv_loss(x_in).sum() * cfg["tv_scale"] + range_loss(x_in).sum() * cfg["range_scale"]
                    + sat_losses.sum() * cfg["sat_scale"])
            if self.lpips_model is not None and self.init is not None and self.init_scale:
                loss = loss + lpips_distance(self.lpips_model, x_in, self.init).sum() * self.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any():
                return torch.zeros_like(x)
            if self.offload:
                memory.park(self.clip_models)
            grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            if self.offload:
                memory.unpark(self.clip_models, device)
        if cfg["clamp_grad"]:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(max=cfg["clamp_max"]) / magnitude
        return grad

    def symmetry(self, x):
        if self.cfg["use_horizontal_symmetry"]:
            w = x.shape[-1]
            x = torch.concat((x[:, :, :, :w // 2], torch.flip(x[:, :, :, :w // 2], [-1])), -1)
        if self.cfg["use_vertical_symmetry"]:
            h = x.shape[-2]
            x = torch.concat((x[:, :, :h // 2, :], torch.flip(x[:, :, :h // 2, :], [-2])), -2)
        return x

    def release(self):
        """The last step's UNet output keeps its whole graph alive until the next step, so let it go."""
        if self.one_pass:
            self.one_pass.leaf = self.one_pass.output = None

    # ---- a picture ----
    def paint(self, start, width, height, skip, batch, stop=None, show=None):
        """One picture at the frame size, through the experiments' loop when any is on. Returns it as a -1 to 1
        tensor. With `stop`, it ends after that many steps and returns its picture at that point, still blurry."""
        show = show or self.to_image
        if self.experiment:
            return lab.denoise(self, start, width, height, skip, batch, stop=stop, show=show)
        return self.denoise(start, width, height, skip, batch, show=show, stop=stop)

    def denoise(self, start, width, height, skip, batch, mask=None, show=None, stop=None):
        """The sampler's own loop for one picture, as paint() describes."""
        cfg, diffusion, hooks, eraser, steer = self.cfg, self.diffusion, self.hooks, self.eraser, self.steer
        show = show or self.to_image
        self.init = self.pull["to"] if self.pull["set"] else start  # Init scale compares against this
        painting_tile = mask is not None
        common = dict(clip_denoised=cfg["clip_denoised"], model_kwargs={}, cond_fn=self.cond_fn, progress=False,
                      skip_timesteps=skip, init_image=start, randomize_class=cfg["randomize_class"])
        self.t = diffusion.num_timesteps - skip - 1
        before, self.ahead = self.ahead, 0
        total_steps = self.t + 1 + before
        shape = (1, 3, height, width)
        common["noise"] = self.start_noise(shape, device)
        if cfg["diffusion_sampling_mode"] == "ddim":
            samples = diffusion.ddim_sample_loop_progressive(
                self.one_pass or self.model, shape, eta=cfg["eta"], transformation_fn=None if painting_tile else self.symmetry,
                transformation_percent=cfg["transformation_percent"], inpainting_mode=painting_tile, mask_inpaint=mask, **common)
        else:
            samples = diffusion.plms_sample_loop_progressive(self.model, shape, order=2, **common)
        for j, sample in enumerate(samples):
            self.t -= 1
            last = self.t == -1 or j + 1 == stop
            if j % cfg["display_rate"] == 0 or last or self.pausing():
                hooks.preview(show(sample["pred_xstart"]))
            if eraser:
                eraser.shown = sample["pred_xstart"]
            hooks.step(j + before, total_steps, batch, self.n_batches, self.seed,  # after the preview, so a pause shows this step
                       at=1 - (self.t + 1) / diffusion.num_timesteps)
            if j + 1 == stop:
                break
            if self.hold_to is not None:  # the sampler reads the sample back for the next step
                sample["sample"] = self.hold_layout(sample["sample"], sample["pred_xstart"], self.t + 1)
            if eraser and not painting_tile and eraser.step(sample["pred_xstart"]):
                self.forget_erased()
                sample["sample"] = eraser.restart(sample["sample"], self.t + 1, self.predict)
            if steer:
                sample["sample"] = steer.pull(sample["sample"], sample["pred_xstart"], self.t + 1, diffusion)
        self.release()
        return sample["pred_xstart"].detach()

    def hand_over(self, start, skip, n, size, noise_seed, keep_as=None):
        """With a handoff model, it paints from `start` (a -1 to 1 tensor, or None for noise), and Disco takes
        its picture. At Handoff 100% the checkpoint finishes it and Disco repaints `skip`'s share, Denoise, as it
        would a start image. Below that it stops partway and Disco carries on from that point. Returns the new
        start and skip. `size` is the size painted when there is no start."""
        cfg, diffusion = self.cfg, self.diffusion
        self.pull["set"], self.pull["to"] = False, None
        if self.handing is None or not self.hand_start:
            return start, skip
        import handoff
        self.release()
        # The checkpoint repaints Start denoise of a start, or with that at 0, the share Disco itself would.
        own = float(cfg.get("handoff_start_denoise") or 0) / 100
        repaints = own or (diffusion.num_timesteps - skip) / diffusion.num_timesteps
        picture = None if start is None else self.to_image(start)
        with handoff.Parked(self.handing, self.resting(), device, *size):
            made = self.handing.paint(picture, repaints, n, *size, noise_seed)
        if made is None:
            return start, skip
        image, alpha = made
        if keep_as:
            image.save(keep_as)
        self.pull["set"], self.pull["to"] = True, start
        if float(cfg["handoff_share"]) < 100:  # it carries on from the point on its own path where the checkpoint stopped
            skip = handoff.disco_skip(diffusion.alphas_cumprod, alpha, skip)
        self.hooks.status(f"Disco carries on from step {skip + 1} of {cfg['steps']}")
        return TF.to_tensor(image).unsqueeze(0).mul(2).sub(1).to(device), skip

    def small_start(self, start, skip, n):
        """Small first: Disco paints the start of the path at `small`, in the frame's shape, and stops at
        small_steps while the picture is still blurry. Enlarged, that picture is the start for the rest of the
        path at full size, where Disco adds the noise for that step as it does to any start image, so it
        carries on from the same point. Returns the new start and skip, like hand_over."""
        cfg, side_x, side_y, small_steps = self.cfg, self.side_x, self.side_y, self.small_steps
        self.hold_to = None
        if not small_steps or skip >= small_steps:
            return start, skip
        width, height = self.small

        def sized(image, w, h):
            if image is None:
                return None
            return F.interpolate(image.float(), size=(h, w), mode="bicubic", antialias=True, align_corners=False)

        held = self.pull["to"] if self.pull["set"] else start  # what Init scale compares against at full size
        self.pull["set"], self.pull["to"] = True, sized(held, width, height)
        self.hooks.status(f"Painting steps {skip + 1} to {small_steps} of {cfg['steps']} at {width}x{height}")
        # The preview is enlarged to the frame, so it looks the same size all the way through.
        rough = self.paint(sized(start, width, height), width, height, skip, n, stop=small_steps - skip,
                           show=lambda image: self.to_image(sized(image, side_x, side_y)))
        self.pull["set"], self.pull["to"] = True, held
        self.ahead = small_steps - skip
        carry_on = small_steps
        repaint = float(cfg["small_denoise"] or 0)
        if repaint > 0:  # the full size repaints its share of the small picture, never less than what is left
            carry_on = min(max(cfg["steps"] - round(cfg["steps"] * repaint / 100), 0), small_steps)
        self.hooks.status(f"Carrying on at {side_x}x{side_y} from step {carry_on + 1} of {cfg['steps']}")
        enlarged = sized(rough, side_x, side_y).clamp(-1, 1)
        if self.hold_curve is not None:
            self.hold_to = enlarged  # Hold the small layout pulls the full size toward this
        return enlarged, carry_on

    def finish(self, image, n, seed_value, keep_as=None):
        """Finish %: the handoff model repaints the end of Disco's finished picture to clean it up."""
        if self.handing is None or not self.hand_finish:
            return image
        import handoff
        self.release()
        if keep_as:
            image.save(keep_as)  # Disco's own picture, before the finish
        with handoff.Parked(self.handing, self.resting(), device, *self.handing.finish_size(*image.size), painted=True):
            return self.handing.finish(image, float(self.cfg["handoff_finish"]) / 100, n, seed_value)

    # ---- saving ----
    def named(self, ending):
        """A file of this run in its folder, such as its settings or a frame: the batch name and run number, then `ending`."""
        return os.path.join(self.folder, f"{self.batch_name}({self.run_number}){ending}")

    def write_notes(self, again=False):
        notes = self.named("_settings.txt")
        if again or not os.path.exists(notes):
            with open(notes, "w", encoding="utf-8") as f:
                json.dump({**self.cfg, "text_prompts": {str(at): lines for at, lines in self.sets}, "seed": self.seed,
                           "init_fit": self.init_fit, "offload": self.offload, "range_scale_works": True,
                           "width_height": [self.side_x, self.side_y], "final_size": list(self.final),
                           **({"small_size": list(self.small), "small_steps": self.small_steps} if self.small else {})}, f, indent=2)

    def save(self, image, batch):
        path = self.named(f"_{batch}.png")
        self.write_notes(again=self.regions["dirty"] and os.path.exists(self.named("_settings.txt")))
        self.regions["dirty"] = False
        image.save(path)
        return path

    def save_sketch(self, sketch):
        self.to_image(sketch).save(self.named("_partner.png"))

    def pausing(self):
        return getattr(self.hooks, "pausing", lambda: False)()

    @staticmethod
    def reseed(seed):
        """Starts every random number generator the render draws from at `seed`, as set_seeds does."""
        set_seeds(seed)

    @staticmethod
    def to_image(tensor):
        return TF.to_pil_image(tensor[0].add(1).div(2).clamp(0, 1).cpu())
