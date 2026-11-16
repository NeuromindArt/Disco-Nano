"""Handoff: a Stable Diffusion checkpoint paints the first part of the path from noise, and Disco carries on.

This file holds HandoffModel, what every handoff model shares, and Handoff for SD 1.5 and SDXL. Z-Image, Flux.1 and
Flux.2 Klein work the same way, through FlowHandoff in flow.py.

The checkpoint (SD 1.5 or SDXL, one .safetensors file) denoises its own latent for its share of the
steps. Its estimate of the finished picture at that point is decoded to pixels, and Disco starts from
the point on its own schedule with the same share of signal, so the two models share one path. With a
start image or an animation frame, the checkpoint starts where Disco would have started and takes its
share of the rest. At 100% no step is left, and Disco only finishes the checkpoint's picture.

The checkpoint waits in RAM between images and frames, and comes to the GPU only for its own steps.
When Disco's models and the checkpoint won't fit together, Disco's wait in RAM instead.
"""
import contextlib
import gc
import time
import types

import torch
from PIL import Image

import budget
import imaging
import memory
from memory import GB


def load(path, status, device):
    """The handoff model for a file: SD 1.5 and SDXL here, Z-Image and Flux in flow.py."""
    import flow
    if path.lower().endswith(".gguf") or flow.model_kind(path):
        return flow.FlowHandoff(path, status, device)
    return Handoff(path, status, device)


def disco_skip(alphas, alpha, least):
    """Disco's skip_steps whose first step has about alpha's share of signal, and at least `least`."""
    steps = len(alphas)
    first = min(range(steps), key=lambda i: abs(float(alphas[i]) - alpha))
    return min(max(steps - 1 - first, least), steps - 1)


SAMPLERS = {  # the Handoff sampler's choices, as diffusers schedulers and their settings
    "Euler": ("EulerDiscreteScheduler", {}),
    "Euler a": ("EulerAncestralDiscreteScheduler", {}),
    "DPM++ 2M": ("DPMSolverMultistepScheduler", {"algorithm_type": "dpmsolver++", "solver_order": 2}),
    "DPM++ 2M SDE": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "solver_order": 2}),
    "DDIM": ("DDIMScheduler", {}),
    "UniPC": ("UniPCMultistepScheduler", {}),
    "LCM": ("LCMScheduler", {}),
}
KARRAS = {"Euler", "DPM++ 2M", "DPM++ 2M SDE", "UniPC"}


def make_scheduler(config, sampler, karras, few_steps):
    """The checkpoint's scheduler for a Handoff sampler, from its own config. Karras applies where the sampler has
    it, and few-step models get trailing spacing."""
    import diffusers
    name, extra = SAMPLERS.get(sampler, SAMPLERS["Euler"])
    if karras and sampler in KARRAS:
        extra = {**extra, "use_karras_sigmas": True}
    if few_steps:
        extra = {**extra, "timestep_spacing": "trailing"}
    return getattr(diffusers, name).from_config(config, **extra)


def size_for(width, height):
    """The checkpoint paints at least 512x512, the smallest size it works well at, and at most 768x768,
    which keeps SDXL inside 8GB. The shape follows the frame, in multiples of 64."""
    area = min(max(width * height, 512 * 512), 768 * 768)
    scale = (area / (width * height)) ** 0.5
    return max(round(width * scale / 64), 1) * 64, max(round(height * scale / 64), 1) * 64


def finish_size(width, height):
    """Where the finish paints: the picture's own shape in multiples of 64, at most about a megapixel."""
    scale = min(1.0, (1024 * 1024 / (width * height)) ** 0.5)
    return max(round(width * scale / 64), 1) * 64, max(round(height * scale / 64), 1) * 64


def on_device(embeds, device):
    """Prompt embeddings, as encode_text() keeps them in RAM, on `device`. Z-Image keeps a list of them."""
    return {k: [v.to(device) for v in e] if isinstance(e, list) else e.to(device) for k, e in embeds.items()}


class HandoffModel:
    """What every handoff model shares. Its text encoders come to the GPU only to encode the prompts, all of them
    before Disco loads. Its painting parts come to the GPU only for its own steps, and go back to RAM after, whatever
    happens. Each step reports, can be stopped, and shows the picture so far. Each kind sets up its parts and paints
    in its own way.

    A kind sets device, encoders (with attach_encoders), vae, unet (the part that paints) and unet_bytes, and has
    to_gpu(on), painters() for its painting parts that go back to RAM, encode_text(), decode() and paint_area()."""
    size_for = staticmethod(size_for)  # where it paints for a frame this size
    finish_size = staticmethod(finish_size)  # where its finish paints

    def __init__(self, device, status):
        self.device, self.status, self.stopped, self.preview = device, status, lambda: False, lambda image: None
        self.embeds, self.next_show = [], 0

    # ---- each render ----
    def prepare(self, cfg, prompt_sets, status, stopped=lambda: False, preview=lambda image: None):
        """Settings and prompts for one render. prompt_sets is [(first frame, [prompt lines])]. Every set is
        encoded now, before Disco loads, so the text encoders can go back to RAM."""
        self.status, self.stopped, self.preview = status, stopped, preview
        self.share = min(max(float(cfg["handoff_share"]) / 100, 0), 1)
        self.steps = max(int(cfg["handoff_steps"]), 1)
        self.guidance = float(cfg["handoff_guidance"])
        self.configure(cfg)
        self.embeds = []
        where = self.encoders_on(status)
        try:
            for at, lines in prompt_sets:
                self.embeds.append((at, self.encode_text(*words(lines, cfg["handoff_negative"]), self.guidance, where)))
        finally:
            self.encoders_off()

    def configure(self, cfg):
        """A kind's own settings for a render."""

    def embeds_for(self, frame):
        """The prompt set in force at `frame`, on the GPU."""
        chosen = next((e for at, e in reversed(self.embeds) if at <= frame), self.embeds[0][1])
        return on_device(chosen, self.device)

    # ---- memory ----
    def parts(self):
        """What goes back to RAM between renders. A part that streams is there already."""
        return [*self.painters(), self.vae, *[e for e in self.encoders if not getattr(e, "streams", False)]]

    def encoders_on(self, status):
        """Brings the text encoders where they can run, and returns that device. On the GPU when they fit with room
        to work. Otherwise each encoder streams there a layer at a time, as a big model does, and if that can't be
        set up, it runs on the CPU, which is slow but always fits."""
        size = sum(p.numel() * p.element_size() for e in self.encoders for p in e.parameters())
        where = self.device
        if memory.room() > size + 0.5 * GB and not any(getattr(e, "streams", False) for e in self.encoders):
            for e in self.encoders:
                e.to(where)
        else:
            try:
                from diffusers.hooks import apply_group_offloading
                for e in self.encoders:
                    if not getattr(e, "streams", False):
                        e.to("cpu")
                        # Layer by layer: an encoder's blocks sit inside it, which block-level streaming doesn't reach,
                        # so it would bring the whole encoder over at once. Each layer goes from the file's own weights
                        # and back to them, so RAM never holds a copy: about 5GB less for Flux's T5.
                        apply_group_offloading(e, onload_device=torch.device(self.device), offload_type="leaf_level",
                                               use_stream=True, low_cpu_mem_usage=True)
                        e.streams = True
            except (ImportError, RuntimeError, ValueError, TypeError, NotImplementedError) as e:
                print(f"Streaming the text encoders didn't set up: {e!r}", flush=True)
                where = "cpu"
                status("Encoding the prompt on the CPU, as the GPU is short of room")
        self.attach_encoders(True)
        return where

    def encoders_off(self):
        self.attach_encoders(False)
        for e in self.encoders:
            if not getattr(e, "streams", False):
                e.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    @contextlib.contextmanager
    def on_gpu(self):
        """The painting parts on the GPU for the model's own steps, and back in RAM after, whatever happened, so
        the GPU goes back to Disco."""
        try:
            self.to_gpu(True)
            yield
        finally:
            self.to_gpu(False)
            gc.collect()
            torch.cuda.empty_cache()

    def painter_off(self):
        """The part that paints leaves the GPU, before the decode, which then has the room."""
        for part in self.painters():
            part.to("cpu")
        torch.cuda.empty_cache()

    # ---- steps ----
    def generator(self, seed):
        return torch.Generator(self.device).manual_seed(int(seed) % 2**32)

    def stepper(self, label, run, show=None, last=False):
        """What each of the model's steps calls back: it says how far it is, stops when asked, and shows the picture
        so far through `show(values)`. With `last`, the model stops after `run` steps, short of its whole path."""
        def check(p, i, t, values):
            self.status(f"{label}, step {i + 1} of {run}")
            if self.stopped():
                raise KeyboardInterrupt
            if show:
                show(values)
            if last and i + 1 >= run:
                p._interrupt = True
            return values
        return check

    def show(self, latent, size, common=None):
        """Sends the picture so far to the preview, decoded as Disco gets it, skipping steps so decoding takes at
        most about a quarter of the time."""
        now = time.monotonic()
        if now < self.next_show:
            return
        self.preview(self.decode(latent, common, preview=True).resize(size, Image.BILINEAR))
        self.next_show = time.monotonic() + 3 * (time.monotonic() - now)

    # ---- Fix's Inpaint ----
    def inpaint_size(self, width, height):
        return self.size_for(width, height)

    @torch.no_grad()
    def inpaint(self, image, mask, text, strength, steps, guidance, seed):
        """The picture with the white of `mask` repainted, both PIL images at a size inpaint_size() gave. `text` may
        be empty: the model then carries on the picture around the area."""
        where = self.encoders_on(self.status)
        try:
            embeds = self.encode_text(text, "", guidance, where)
        finally:
            self.encoders_off()
        check = self.stepper("Inpainting", max(int(steps * strength), 1))
        with self.on_gpu():
            return self.paint_area(image, mask, on_device(embeds, self.device), strength, steps, guidance, seed, check)


class Handoff(HandoffModel):
    """SD 1.5 and SDXL."""

    def __init__(self, path, status, device):
        """Loads the checkpoint into RAM. The render worker keeps it for later renders, so this happens once
        until the file or the app changes. prepare() sets it up for each render."""
        import diffusers
        diffusers.utils.logging.set_verbosity_error()
        super().__init__(device, status)
        about = budget.describe_checkpoint(path) or {}
        self.xl, self.few_steps, self.clip_skip = about.get("kind") == "sdxl", bool(about.get("few_steps")), 0
        status(f"Loading the {'SDXL' if self.xl else 'SD 1.5'} handoff model")
        kind = diffusers.StableDiffusionXLPipeline if self.xl else diffusers.StableDiffusionPipeline
        extra = {} if self.xl else {"safety_checker": None, "requires_safety_checker": False}
        pipe = kind.from_single_file(path, torch_dtype=torch.float16, **extra)
        self.scheduler_config = pipe.scheduler.config
        extra_image = {} if self.xl else {"requires_safety_checker": False}
        image_kind = diffusers.StableDiffusionXLImg2ImgPipeline if self.xl else diffusers.StableDiffusionImg2ImgPipeline
        # Built from the same parts. from_pipe would turn SDXL's norm layers to float32.
        self.text_pipe, self.image_pipe = pipe, image_kind(**pipe.components, **extra_image)
        # The text encoders wait in RAM, outside the pipelines, which then take the device from the parts that run.
        self.encoders = [pipe.text_encoder] + ([pipe.text_encoder_2] if self.xl else [])
        for e in self.encoders:
            # Clip skip in diffusers reaches the final norm through text_model, which transformers 5 folded into the
            # model itself. A plain attribute, kept out of the module tree, so the model isn't listed inside itself.
            if not hasattr(e, "text_model") and hasattr(e, "final_layer_norm"):
                object.__setattr__(e, "text_model", types.SimpleNamespace(final_layer_norm=e.final_layer_norm))
        self.attach_encoders(False)
        self.unet, self.vae = pipe.unet.eval(), pipe.vae.eval()
        if self.xl and torch.cuda.get_device_properties(device).total_memory < 12 * GB:
            # SDXL's UNet is 5.1GB in float16, too much beside anything else on 8GB. Stored in 8 bits and
            # computed in 16 it takes 2.6GB. Disco repaints what follows, which hides the small loss.
            self.unet.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16)
        self.vae.enable_tiling()  # decoding at 768x768 in float32 is the biggest moment otherwise
        self.unet_bytes = sum(p.numel() * p.element_size() for p in self.unet.parameters())

    def attach_encoders(self, on):
        for p in (self.text_pipe, self.image_pipe):
            p.text_encoder = self.encoders[0] if on else None
            if self.xl:
                p.text_encoder_2 = self.encoders[1] if on else None

    def painters(self):
        return [self.unet]

    def to_gpu(self, on):
        for part in (self.vae, self.unet):
            part.to(self.device if on else "cpu")

    def configure(self, cfg):
        self.clip_skip = max(int(cfg.get("handoff_clip_skip") or 1), 1) - 1  # diffusers counts the layers left out
        self.text_pipe.scheduler = self.image_pipe.scheduler = make_scheduler(
            self.scheduler_config, cfg.get("handoff_sampler", "Euler"), bool(cfg.get("handoff_karras")), bool(cfg["handoff_few_steps"]))

    @torch.no_grad()
    def encode_text(self, text, negative, guidance, where):
        """One prompt's embeddings, in RAM, as the pipelines take them, for this guidance."""
        names = (["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]
                 if self.xl else ["prompt_embeds", "negative_prompt_embeds"])
        both = {"prompt_2": text, "negative_prompt_2": negative} if self.xl else {}
        out = self.text_pipe.encode_prompt(prompt=text, device=where, num_images_per_prompt=1,
                                           do_classifier_free_guidance=guidance > 1, negative_prompt=negative,
                                           clip_skip=self.clip_skip or None, **both)
        return {k: v.cpu() for k, v in zip(names, out) if v is not None}

    def needs_room(self, width, height, painted=False):
        """Bytes the checkpoint needs on the GPU, with its working memory, for a frame this size. painted means
        it paints at exactly this size, as the finish does."""
        if not painted:
            width, height = size_for(width, height)
        working = (1.3 if self.xl else 0.8) * GB * (width * height) / (512 * 512)
        return self.unet_bytes + 0.5 * GB + working

    @torch.no_grad()
    def paint(self, start, repaints, frame, width, height, seed):
        """The checkpoint's picture estimate where it hands over, as an image the size of `start` or
        width x height, and that point's share of signal. None when its share comes to no steps.

        start is a PIL image or None, and `repaints` the share of it the checkpoint repaints, like Denoise."""
        sd_w, sd_h = size_for(width, height)
        scheduler = self.text_pipe.scheduler
        scheduler.set_timesteps(self.steps)
        times = [int(t) for t in scheduler.timesteps]
        alphas = scheduler.alphas_cumprod
        share = min(max(float(repaints), 0), 1)
        begin = 0 if start is None else min(max(len(times) - round(share * len(times)), 0), len(times) - 1)
        run = round(self.share * (len(times) - begin))
        if run < 1:
            return None
        shown = start.size if start is not None else (width, height)
        seen, self.next_show = {}, 0
        with self.on_gpu():
            common = dict(num_inference_steps=self.steps, guidance_scale=self.guidance, generator=self.generator(seed),
                          output_type="latent", **self.embeds_for(frame))
            if start is None:
                pipe = self.text_pipe
                common.update(width=sd_w, height=sd_h)
            else:
                pipe = self.image_pipe
                latent = self.encode(start.convert("RGB").resize((sd_w, sd_h), Image.LANCZOS))
                common.update(image=latent, strength=min((len(times) - begin + 0.5) / len(times), 1.0))
            step = pipe.scheduler.step

            def keep(*a, **k):
                out = step(*a, **k)
                # The picture estimate: Euler, Euler a, DDIM and LCM return it, and the multistep samplers keep it.
                parts = out if isinstance(out, tuple) else tuple(out.values())  # the pipeline asks for a tuple
                seen["x0"] = parts[1] if len(parts) > 1 else pipe.scheduler.model_outputs[-1]
                seen["t"] = int(a[1] if len(a) > 1 else k["timestep"])
                return out

            # its picture so far, as Disco's preview shows its own
            check = self.stepper("Handoff model", run, lambda values: "x0" in seen and self.show(seen["x0"], shown), last=True)
            pipe.scheduler.step = keep
            try:
                pipe(callback_on_step_end=check, **common)
            finally:
                pipe.scheduler.__dict__.pop("step", None)
            # Disco carries on from the point after the checkpoint's last step. Not every sampler counts its steps.
            path = [int(t) for t in pipe.scheduler.timesteps]
            after = path.index(seen["t"]) + 1 if seen["t"] in path else len(path)
            alpha = float(alphas[path[after]]) if after < len(path) else 1.0
            self.painter_off()
            picture = self.decode(seen.pop("x0"))
            self.preview(picture.resize(shown, Image.BILINEAR))  # exactly what Disco gets
            seen.clear()
            common = None
        return picture.resize(shown, Image.LANCZOS), alpha

    @torch.no_grad()
    def finish(self, image, strength, frame, seed):
        """Finish: repaints the last `strength` share of the path from Disco's finished picture, so the
        checkpoint cleans up what Disco made. It works at the picture's own size, up to about a megapixel, and
        returns a picture the size it was given."""
        width, height = image.size
        sd_w, sd_h = finish_size(width, height)
        run = max(round(self.steps * strength), 1)
        self.next_show = 0
        with self.on_gpu():
            latent = self.encode(image.convert("RGB").resize((sd_w, sd_h), Image.LANCZOS))
            check = self.stepper("Handoff model finishing", run, lambda values: self.show(values["latents"], (width, height)))
            # diffusers runs int(steps * strength) steps; half a step more makes that `run`, never 0 with few steps
            out = self.image_pipe(image=latent, strength=min((run + 0.5) / self.steps, 1.0), num_inference_steps=self.steps,
                                  guidance_scale=self.guidance, generator=self.generator(seed), output_type="latent",
                                  callback_on_step_end=check, **self.embeds_for(frame))
            self.painter_off()
            picture = self.decode(out.images)
            out = None
        return picture.resize((width, height), Image.LANCZOS)

    def inpaint_pipe(self):
        """diffusers' inpainting pipeline for this checkpoint, built once from the parts already loaded, with an Euler
        scheduler, spaced for few-step models when this is one. It paints the masked part from `strength` of the way
        to noise, and at each step puts back the rest of the picture at that step's noise level, so the two meet
        without a seam."""
        import diffusers
        if getattr(self, "_inpaint_pipe", None) is None:
            parts = dict(self.text_pipe.components, text_encoder=None)
            if self.xl:
                pipe = diffusers.StableDiffusionXLInpaintPipeline(**{**parts, "text_encoder_2": None})
            else:
                pipe = diffusers.StableDiffusionInpaintPipeline(**{**parts, "safety_checker": None}, requires_safety_checker=False)
            pipe.set_progress_bar_config(disable=True)
            pipe.scheduler = make_scheduler(self.scheduler_config, "Euler", False, self.few_steps)
            self._inpaint_pipe = pipe
        return self._inpaint_pipe

    def paint_area(self, image, mask, embeds, strength, steps, guidance, seed, check):
        out = self.inpaint_pipe()(image=image, mask_image=mask, strength=strength, num_inference_steps=steps, height=image.height,
                                  width=image.width, guidance_scale=guidance, callback_on_step_end=check,
                                  generator=self.generator(seed), **embeds)
        return out.images[0]

    @contextlib.contextmanager
    def vae_in_float(self, wanted=True):
        """The VAE in float32 for a moment, as SDXL's needs for encoding and both need for decoding, since it
        overflows in float16. It goes back to float16 whatever happens, so one failed decode can't leave it in
        float32, at twice the memory, for every render after."""
        if not wanted:
            yield self.vae
            return
        try:
            yield self.vae.float()
        finally:
            self.vae.half()

    def encode(self, image):
        with self.vae_in_float(self.xl) as vae:
            pixels = imaging.from_pil(image, device=self.device, signed=True).to(vae.dtype)
            latent = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor
        return latent.half()

    def decode(self, latent, common=None, preview=False):
        """Pixels from a latent. For the preview, while the UNet is on the GPU, it decodes in full, as a shrunken
        latent decodes to blur and stripes, in small tiles."""
        with self.vae_in_float() as vae:
            wide = vae.tile_sample_min_size, vae.tile_latent_min_size
            if preview:
                vae.tile_sample_min_size, vae.tile_latent_min_size = 512, 64
            try:
                picture = vae.decode(latent.float() / vae.config.scaling_factor).sample
            finally:
                vae.tile_sample_min_size, vae.tile_latent_min_size = wide
        return imaging.to_pil(picture, signed=True)


def words(lines, negative):
    """The checkpoint's prompt from Disco's prompt lines. A line with a negative weight steers away."""
    from prompts import parse_prompt
    parsed = [parse_prompt(line) for line in lines]
    away = [text for text, weight in parsed if weight < 0]
    return ", ".join(text for text, weight in parsed if weight > 0), ", ".join([negative] * bool(negative) + away)


class Parked:
    """Moves Disco's models to RAM while the checkpoint has the GPU, when the two won't fit together."""

    def __init__(self, handoff, modules, device, width, height, painted=False):
        self.handoff, self.modules, self.device = handoff, [m for m in modules if m is not None], device
        self.size, self.painted, self.moved = (width, height), painted, False

    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        room = memory.room()
        if room < self.handoff.needs_room(*self.size, painted=self.painted):
            self.handoff.status("Moving Disco to RAM while the handoff model paints")
            for module in self.modules:
                module.to("cpu")
            self.moved = True
            torch.cuda.empty_cache()
        return self

    def __exit__(self, *exc):
        if self.moved:
            for module in self.modules:
                module.to(self.device)
        return False
