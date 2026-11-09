"""A small text model that works beside Disco during a render.

It denoises its own copy of the picture in latent space, and Disco borrows its self-attention. Those
maps say which parts of the picture belong together. They carry no color or texture, and they are
the same shape in both models, so they need no training or adapter.
"""
import math
import os

import torch
from torch import nn

import models
import paths

SD15_UNET = {
    "sample_size": 64, "in_channels": 4, "out_channels": 4, "layers_per_block": 2, "cross_attention_dim": 768,
    "attention_head_dim": 8, "block_out_channels": (320, 640, 1280, 1280),
    "down_block_types": ("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
    "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
}


def load_packages():
    try:
        import diffusers
        import transformers
        return diffusers, transformers
    except ImportError as e:
        raise ValueError("The partner model needs the diffusers package. Run setup again to add it.") from e


# TAESD, by Ollin Boer Bohan (MIT): a very small stand-in for the Stable Diffusion VAE.
def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 2) * 2


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def taesd_encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4))


def taesd_decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3))


class Capture:
    """An attention processor that also keeps the self-attention maps."""

    def __init__(self, keep):
        self.keep, self.on, self.maps, self.counts = keep, False, {}, {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **_):
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key = attn.head_to_batch_dim(attn.to_k(context))
        value = attn.head_to_batch_dim(attn.to_v(context))
        probs = attn.get_attention_scores(query, key, None)
        length = probs.shape[1]
        if self.on and encoder_hidden_states is None and length in self.keep:
            average = probs.float().mean(0)
            self.maps[length] = self.maps.get(length, 0) + average
            self.counts[length] = self.counts.get(length, 0) + 1
        out = attn.batch_to_head_dim(torch.bmm(probs, value))
        return attn.to_out[1](attn.to_out[0](out))

    def take(self):
        maps = {k: (v / self.counts[k]).half() for k, v in self.maps.items()}
        self.maps, self.counts = {}, {}
        return maps


class Partner:
    GUIDANCE = 7.5
    WARMUP_STEPS = 10

    def __init__(self, source, file, prompt, height, width, follow, device, status=print):
        diffusers, transformers = load_packages()
        self.device, self.follow = device, follow
        self.strength, self.maps = 0.0, {}
        self.seen, self.difference, self.watching = {}, {}, False
        self.height, self.width = height, width
        models.ensure(models.PARTNER, status)  # an SD 1.5 file still uses Tiny SD's tokenizer and text config
        folder = os.path.join(paths.MODELS, models.PARTNER)
        status("Loading the partner model")
        if source == "Tiny SD":
            self.unet = diffusers.UNet2DConditionModel.from_pretrained(folder, subfolder="unet", torch_dtype=torch.float16)
            text_encoder = transformers.CLIPTextModel.from_pretrained(folder, subfolder="text_encoder", torch_dtype=torch.float16)
        else:
            if not os.path.isfile(file):
                raise ValueError("Set the SD 1.5 file under Experimental in All settings.")
            self.unet, text_encoder = self.load_file(file, folder, diffusers, transformers)
        tokenizer = transformers.CLIPTokenizer.from_pretrained(folder, subfolder="tokenizer")
        self.unet.requires_grad_(False).eval().to(device)

        def encode(text):
            tokens = tokenizer(text, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
            return text_encoder(tokens.input_ids)[0].to(device)

        with torch.no_grad():
            self.text, self.empty = encode(prompt), encode("")
        del text_encoder, tokenizer

        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
        self.alphas = torch.cumprod(1 - betas, 0).to(device).float()
        self.encoder = self.load_taesd(taesd_encoder(), "taesd_encoder", status)
        self.decoder = self.load_taesd(taesd_decoder(), "taesd_decoder", status)
        self.restart(0)
        # Disco has attention at 1/16, 1/32 and 1/64 of the image size.
        self.capture = Capture({(height // s) * (width // s) for s in (16, 32, 64)})
        self.unet.set_attn_processor(self.capture)

    def restart(self, seed):
        """Starts a new picture. It has its own random numbers, so it never changes Disco's."""
        generator = torch.Generator(self.device).manual_seed(seed)
        self.noise = torch.randn(1, 4, self.height // 8, self.width // 8, device=self.device, generator=generator).half()
        self.z0, self.eps = None, None

    def to(self, device):
        for network in (self.unet, self.encoder, self.decoder):
            network.to(device)

    def load_file(self, path, folder, diffusers, transformers):
        from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint
        from safetensors import safe_open
        unet_state, text_state = {}, {}
        prefix = "cond_stage_model.transformer."
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.startswith("model.diffusion_model."):
                    unet_state[key] = f.get_tensor(key).half()
                elif key.startswith(prefix):
                    text_state[key[len(prefix):]] = f.get_tensor(key).half()
        unet = diffusers.UNet2DConditionModel(**SD15_UNET).half()
        unet.load_state_dict(convert_ldm_unet_checkpoint(unet_state, dict(unet.config)))
        config = transformers.CLIPTextConfig.from_pretrained(folder, subfolder="text_encoder")
        text_encoder = transformers.CLIPTextModel(config).half()
        text_encoder.load_state_dict(text_state, strict=False)
        return unet, text_encoder

    def load_taesd(self, network, name, status):
        network.load_state_dict(torch.load(models.ensure(name, status), map_location="cpu"))
        return network.half().requires_grad_(False).eval().to(self.device)

    def watch(self, blocks):
        """Starts recording what the prompt changes at these blocks."""
        def keeper(grid):
            def keep(_, __, output):
                self.seen[grid] = output[0]
            return keep

        self.watching = True
        for grid, block in blocks.items():
            block.register_forward_hook(keeper(grid))

    def predict(self, z, t, keep_maps):
        timestep = torch.tensor([t], device=self.device)
        self.capture.on = keep_maps
        cond = self.unet(z, timestep, encoder_hidden_states=self.text).sample
        with_prompt = dict(self.seen)
        self.capture.on = False
        uncond = self.unet(z, timestep, encoder_hidden_states=self.empty).sample
        if self.watching and keep_maps:
            self.difference = {grid: with_prompt[grid] - self.seen[grid] for grid in self.seen}
        return uncond + self.GUIDANCE * (cond - uncond)

    def step(self, z0, eps, t, keep_maps=True):
        """One move along the partner's own path: DDIM to noise level t, then a new prediction there."""
        alpha = self.alphas[t]
        z = self.noise * 1.0 if z0 is None else alpha.sqrt() * z0 + (1 - alpha).sqrt() * eps
        eps = self.predict(z, t, keep_maps)
        return (z - (1 - alpha).sqrt() * eps) / alpha.sqrt(), eps

    def nearest(self, alpha_bar):
        return int((self.alphas - alpha_bar).abs().argmin())

    @torch.no_grad()
    def refresh(self, image, alpha_bar):
        """Moves the partner to Disco's noise level and collects new maps."""
        t = self.nearest(alpha_bar)
        if self.z0 is None:
            # Disco can start partway down its schedule, but the partner has to start from pure noise.
            for s in torch.linspace(999, t, self.WARMUP_STEPS).round().int().tolist()[:-1]:
                self.z0, self.eps = self.step(self.z0, self.eps, s, keep_maps=False)
        elif self.follow:
            painted = self.encoder(image.add(1).div(2).clamp(0, 1).half())
            self.z0 = torch.lerp(self.z0, painted, self.follow)
        self.z0, self.eps = self.step(self.z0, self.eps, t)
        self.maps = self.capture.take()

    @torch.no_grad()
    def paint(self, steps):
        """Paints a whole picture alone, for a render that only needs the finished sketch."""
        for t in torch.linspace(999, 0, steps).round().int().tolist():
            self.z0, self.eps = self.step(self.z0, self.eps, t, keep_maps=False)
        return self.sketch()

    @torch.no_grad()
    def sketch(self):
        """What the partner has been painting, for comparison."""
        return self.decoder(self.z0).float().clamp(0, 1).mul(2).sub(1)


def install(partner):
    """Blends the partner's maps into Disco's attention blocks."""
    from guided_diffusion.unet import QKVAttentionLegacy

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        kind = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1)
        borrowed = partner.maps.get(length) if partner.strength else None
        if borrowed is not None:
            weight = torch.lerp(weight, borrowed.float(), partner.strength)
        a = torch.einsum("bts,bcs->bct", weight.type(kind), v)
        return a.reshape(bs, -1, length)

    QKVAttentionLegacy.forward = forward
