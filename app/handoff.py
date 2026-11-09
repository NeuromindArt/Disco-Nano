"""Handoff: a Stable Diffusion checkpoint paints the first part of the path from noise, and Disco carries on.

The checkpoint (SD 1.5 or SDXL, one .safetensors file) denoises its own latent for its share of the
steps. Its estimate of the finished picture at that point is decoded to pixels, and Disco starts from
the point on its own schedule with the same share of signal, so the two models share one path. With a
start image or an animation frame, the checkpoint starts where Disco would have started and takes its
share of the rest. At 100% no step is left, and Disco only finishes the checkpoint's picture.

The checkpoint waits in RAM between images and frames, and comes to the GPU only for its own steps.
When Disco's models and the checkpoint won't fit together, Disco's wait in RAM instead.
"""
import gc

import torch
from PIL import Image

import memory

GB = 2**30


def is_xl(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return any(k.startswith("conditioner.embedders.1") for k in f.keys())


def disco_skip(alphas, alpha, least):
    """Disco's skip_steps whose first step has about alpha's share of signal, and at least `least`."""
    steps = len(alphas)
    first = min(range(steps), key=lambda i: abs(float(alphas[i]) - alpha))
    return min(max(steps - 1 - first, least), steps - 1)


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


class Handoff:
    def __init__(self, path, status, device):
        """Loads the checkpoint into RAM. The render worker keeps it for later renders, so this happens once
        until the file or the app changes. prepare() sets it up for each render."""
        import diffusers
        diffusers.utils.logging.set_verbosity_error()
        self.xl, self.device, self.status, self.stopped = is_xl(path), device, status, lambda: False
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

    def parts(self):
        """Everything that goes back to RAM between renders."""
        return [self.unet, self.vae, *self.encoders]

    def prepare(self, cfg, prompt_sets, status, stopped=lambda: False):
        """Settings and prompts for one render. prompt_sets is [(first frame, [prompt lines])]. Every set is
        encoded now, before Disco loads, so the text encoders can go back to RAM."""
        import diffusers
        self.status, self.stopped = status, stopped
        self.share = min(max(float(cfg["handoff_share"]) / 100, 0), 1)
        self.steps = max(int(cfg["handoff_steps"]), 1)
        self.guidance = float(cfg["handoff_guidance"])
        spacing = {"timestep_spacing": "trailing"} if cfg["handoff_few_steps"] else {}
        scheduler = diffusers.EulerDiscreteScheduler.from_config(self.scheduler_config, **spacing)
        self.text_pipe.scheduler = self.image_pipe.scheduler = scheduler
        for e in self.encoders:
            e.to(self.device)
        self.attach_encoders(True)
        self.embeds = []
        names = (["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds", "negative_pooled_prompt_embeds"]
                 if self.xl else ["prompt_embeds", "negative_prompt_embeds"])
        try:
            with torch.no_grad():
                for at, lines in prompt_sets:
                    text, negative = words(lines, cfg["handoff_negative"])
                    both = {"prompt_2": text, "negative_prompt_2": negative} if self.xl else {}
                    out = self.text_pipe.encode_prompt(prompt=text, device=self.device, num_images_per_prompt=1,
                                                       do_classifier_free_guidance=self.guidance > 1, negative_prompt=negative, **both)
                    self.embeds.append((at, {k: v.cpu() for k, v in zip(names, out) if v is not None}))
        finally:
            self.attach_encoders(False)
            for e in self.encoders:
                e.to("cpu")
            out = None
            gc.collect()
            torch.cuda.empty_cache()

    def needs_room(self, width, height, painted=False):
        """Bytes the checkpoint needs on the GPU, with its working memory, for a frame this size. painted means
        it paints at exactly this size, as the finish does."""
        if not painted:
            width, height = size_for(width, height)
        working = (1.3 if self.xl else 0.8) * GB * (width * height) / (512 * 512)
        return self.unet_bytes + 0.5 * GB + working

    @torch.no_grad()
    def paint(self, start, start_alpha, frame, width, height, seed):
        """The checkpoint's picture estimate where it hands over, as an image the size of `start` or
        width x height, and that point's share of signal. None when its share comes to no steps.

        start is a PIL image or None. start_alpha is the share of signal Disco would have started at."""
        sd_w, sd_h = size_for(width, height)
        scheduler = self.text_pipe.scheduler
        scheduler.set_timesteps(self.steps)
        times = [int(t) for t in scheduler.timesteps]
        alphas = scheduler.alphas_cumprod
        if start is None:
            begin = 0
        else:
            begin = min(range(len(times)), key=lambda i: abs(float(alphas[times[i]]) - start_alpha))
        run = round(self.share * (len(times) - begin))
        if run < 1:
            return None
        embeds = next((e for at, e in reversed(self.embeds) if at <= frame), self.embeds[0][1])
        seen = {}
        try:
            embeds = {k: v.to(self.device) for k, v in embeds.items()}
            self.vae.to(self.device)
            self.unet.to(self.device)
            generator = torch.Generator(self.device).manual_seed(int(seed) % 2**32)
            common = dict(num_inference_steps=self.steps, guidance_scale=self.guidance, generator=generator,
                          output_type="latent", **embeds)
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
                seen["x0"] = out[1] if isinstance(out, tuple) else out.pred_original_sample  # the pipeline asks for a tuple
                return out

            def check(p, i, t, values):
                self.status(f"Handoff model, step {i + 1} of {run}")
                if self.stopped():
                    raise KeyboardInterrupt
                if i + 1 >= run:
                    p._interrupt = True
                return values

            pipe.scheduler.step = keep
            pipe(callback_on_step_end=check, **common)
            index = pipe.scheduler.step_index
            alpha = float(alphas[int(pipe.scheduler.timesteps[index])]) if index < len(pipe.scheduler.timesteps) else 1.0
            self.unet.to("cpu")
            torch.cuda.empty_cache()
            picture = self.decode(seen.pop("x0"))
        finally:  # whatever happened, the GPU goes back to Disco
            for p in (self.text_pipe, self.image_pipe):
                p.scheduler.__dict__.pop("step", None)
            self.unet.to("cpu")
            self.vae.to("cpu")
            seen.clear()
            embeds = common = generator = None
        gc.collect()
        torch.cuda.empty_cache()
        return picture.resize(start.size if start is not None else (width, height), Image.LANCZOS), alpha

    @torch.no_grad()
    def finish(self, image, strength, frame, seed):
        """Finish: repaints the last `strength` share of the path from Disco's finished picture, so the
        checkpoint cleans up what Disco made. It works at the picture's own size, up to about a megapixel, and
        returns a picture the size it was given."""
        width, height = image.size
        sd_w, sd_h = finish_size(width, height)
        run = max(round(self.steps * strength), 1)
        embeds = next((e for at, e in reversed(self.embeds) if at <= frame), self.embeds[0][1])
        try:
            embeds = {k: v.to(self.device) for k, v in embeds.items()}
            self.vae.to(self.device)
            self.unet.to(self.device)
            latent = self.encode(image.convert("RGB").resize((sd_w, sd_h), Image.LANCZOS))
            generator = torch.Generator(self.device).manual_seed(int(seed) % 2**32)

            def check(p, i, t, values):
                self.status(f"Handoff model finishing, step {i + 1} of {run}")
                if self.stopped():
                    raise KeyboardInterrupt
                return values

            out = self.image_pipe(image=latent, strength=min(max(strength, 0.01), 1.0), num_inference_steps=self.steps,
                                  guidance_scale=self.guidance, generator=generator, output_type="latent",
                                  callback_on_step_end=check, **embeds)
            self.unet.to("cpu")
            torch.cuda.empty_cache()
            picture = self.decode(out.images)
        finally:  # whatever happened, the GPU goes back to Disco
            self.unet.to("cpu")
            self.vae.to("cpu")
            embeds = out = generator = None
        gc.collect()
        torch.cuda.empty_cache()
        return picture.resize((width, height), Image.LANCZOS)

    def encode(self, image):
        vae = self.vae.float() if self.xl else self.vae  # SDXL's VAE overflows in float16
        pixels = torch.from_numpy(to_pixels(image)).to(self.device, vae.dtype)
        latent = vae.encode(pixels).latent_dist.mean * vae.config.scaling_factor
        self.vae.half()
        return latent.half()

    def decode(self, latent):
        vae = self.vae.float()
        picture = vae.decode(latent.float() / vae.config.scaling_factor).sample.clamp(-1, 1)
        self.vae.half()
        return Image.fromarray(picture[0].add(1).mul(127.5).round().byte().permute(1, 2, 0).cpu().numpy())


def to_pixels(image):
    import numpy as np
    return (np.asarray(image, dtype=np.float32) / 127.5 - 1).transpose(2, 0, 1)[None].copy()


def words(lines, negative):
    """The checkpoint's prompt from Disco's prompt lines. A line with a negative weight steers away."""
    from disco import parse_prompt
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
