"""Handoff with flow models: Z-Image, Flux.1 and Flux.2 Klein.

These come as separate files, the way ComfyUI keeps them: the model itself (.safetensors or .gguf), its text
encoder (Qwen3-4B for Z-Image and Klein 4B, Qwen3-8B for Klein 9B, T5-XXL and CLIP-L for Flux) and an autoencoder,
Flux's, which Z-Image shares, or Flux.2's.
The text encoder and autoencoder are found beside the model, in the same models folder, by what is inside
each file rather than by its name. Their configs and tokenizers are small downloads the first time.

They paint a straight line from noise to the picture instead of SD's curve, so a step's noise level is a
sigma from 1 to 0 rather than a share of signal. The picture estimate at any step is the latent minus sigma
times what the model predicts, and Disco carries on from the point with the same balance of signal and noise.

Big weights stay in 8 bits in memory and are cast to 16 per layer as they compute, as ComfyUI does: fp8 files
as they are, bf16 ones on loading, and GGUF files by diffusers. ComfyUI's mixed files, where a few layers are
NVFP4, are unpacked by the published format: high nibble first, block scales in the tensor-core tile order.
"""
import contextlib
import gc
import json
import os
import time

import torch
from PIL import Image

import flowfiles
import imaging
import memory
import models
import paths
from flowfiles import KINDS, describe, find_parts, header, missing_parts, model_kind, variant
from handoff import HandoffModel
from memory import GB

COMPUTE = torch.bfloat16
FP8 = torch.float8_e4m3fn
SUPPORT = os.path.join(paths.MODELS, "handoff-support")


# --------------------------------------------------------------------------------- configs and tokenizers

FLUX_TRANSFORMER = {"_class_name": "FluxTransformer2DModel", "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56],
                    "in_channels": 64, "joint_attention_dim": 4096, "num_attention_heads": 24, "num_layers": 19,
                    "num_single_layers": 38, "patch_size": 1, "pooled_projection_dim": 768}
FLUX_SCHEDULER = {"dev": {"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 0.5,
                          "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 3.0,
                          "use_dynamic_shifting": True},
                  "schnell": {"_class_name": "FlowMatchEulerDiscreteScheduler", "num_train_timesteps": 1000, "shift": 1.0,
                              "use_dynamic_shifting": False}}
VAE16 = {"_class_name": "AutoencoderKL", "act_fn": "silu", "block_out_channels": [128, 256, 512, 512],
         "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True, "in_channels": 3, "latent_channels": 16,
         "layers_per_block": 2, "mid_block_add_attention": True, "norm_num_groups": 32, "out_channels": 3,
         "sample_size": 1024, "scaling_factor": 0.3611, "shift_factor": 0.1159, "up_block_types": ["UpDecoderBlock2D"] * 4,
         "use_post_quant_conv": False, "use_quant_conv": False}
CLIP_L = {"architectures": ["CLIPTextModel"], "attention_dropout": 0.0, "bos_token_id": 0, "dropout": 0.0, "eos_token_id": 2,
          "hidden_act": "quick_gelu", "hidden_size": 768, "initializer_factor": 1.0, "initializer_range": 0.02,
          "intermediate_size": 3072, "layer_norm_eps": 1e-05, "max_position_embeddings": 77, "model_type": "clip_text_model",
          "num_attention_heads": 12, "num_hidden_layers": 12, "pad_token_id": 1, "projection_dim": 768, "vocab_size": 49408}
T5_XXL = {"architectures": ["T5EncoderModel"], "d_ff": 10240, "d_kv": 64, "d_model": 4096, "decoder_start_token_id": 0,
          "dense_act_fn": "gelu_new", "dropout_rate": 0.1, "eos_token_id": 1, "feed_forward_proj": "gated-gelu",
          "initializer_factor": 1.0, "is_encoder_decoder": True, "is_gated_act": True, "layer_norm_epsilon": 1e-06,
          "model_type": "t5", "num_decoder_layers": 24, "num_heads": 64, "num_layers": 24, "pad_token_id": 0,
          "relative_attention_max_distance": 128, "relative_attention_num_buckets": 32, "tie_word_embeddings": False,
          "vocab_size": 32128}
# Flux.2 Klein, as Black Forest Labs publishes it. Klein has no guidance embedding: the distilled builds take no
# guidance, and the Base builds take real CFG.
FLUX2_TRANSFORMER = {size: {"_class_name": "Flux2Transformer2DModel", "attention_head_dim": 128, "axes_dims_rope": [32, 32, 32, 32],
                            "eps": 1e-06, "guidance_embeds": False, "in_channels": 128, "mlp_ratio": 3.0, "out_channels": None,
                            "patch_size": 1, "rope_theta": 2000, "timestep_guidance_channels": 256, **own}
                     for size, own in {"4b": {"joint_attention_dim": 7680, "num_attention_heads": 24, "num_layers": 5, "num_single_layers": 20},
                                       "9b": {"joint_attention_dim": 12288, "num_attention_heads": 32, "num_layers": 8, "num_single_layers": 24}}.items()}
FLUX2_VAE = {"_class_name": "AutoencoderKLFlux2", "act_fn": "silu", "batch_norm_eps": 0.0001, "batch_norm_momentum": 0.1,
             "block_out_channels": [128, 256, 512, 512], "down_block_types": ["DownEncoderBlock2D"] * 4, "force_upcast": True,
             "in_channels": 3, "latent_channels": 32, "layers_per_block": 2, "mid_block_add_attention": True, "norm_num_groups": 32,
             "out_channels": 3, "patch_size": [2, 2], "sample_size": 1024, "up_block_types": ["UpDecoderBlock2D"] * 4,
             "use_post_quant_conv": True, "use_quant_conv": True}
FLUX2_SCHEDULER = {"_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 0.5,
                   "max_image_seq_len": 4096, "max_shift": 1.15, "num_train_timesteps": 1000, "shift": 3.0,
                   "time_shift_type": "exponential", "use_dynamic_shifting": True}
# Qwen3-4B and Qwen3-8B differ in width only. Klein reads three of their hidden layers.
QWEN3 = {size: {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": False, "attention_dropout": 0.0,
                "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 128, "hidden_act": "silu", "initializer_range": 0.02,
                "max_position_embeddings": 40960, "max_window_layers": 36, "num_attention_heads": 32, "num_hidden_layers": 36,
                "num_key_value_heads": 8, "rms_norm_eps": 1e-06, "rope_scaling": None, "rope_theta": 1000000, "sliding_window": None,
                "use_cache": True, "use_sliding_window": False, "vocab_size": 151936, **own}
         for size, own in {"4b": {"hidden_size": 2560, "intermediate_size": 9728, "tie_word_embeddings": True},
                           "9b": {"hidden_size": 4096, "intermediate_size": 12288, "tie_word_embeddings": False}}.items()}


def support(kind, status):
    """The folder with `kind`'s configs and tokenizers, downloading them the first time. Klein reads its prompt
    with the same Qwen3 tokenizer as Z-Image, so it shares that download."""
    folder = os.path.join(SUPPORT, kind)
    for name in models.FLOW_SUPPORT[kind]:
        models.ensure(name, status)
    if kind == "flux2":
        write(os.path.join(folder, "vae", "config.json"), FLUX2_VAE)
        for size, config in QWEN3.items():
            write(os.path.join(folder, f"text_encoder_{size}", "config.json"), config)
        return folder
    write(os.path.join(folder, "vae", "config.json"), VAE16)
    if kind == "flux":
        write(os.path.join(folder, "text_encoder", "config.json"), CLIP_L)
        write(os.path.join(folder, "text_encoder_2", "config.json"), T5_XXL)
        write(os.path.join(folder, "tokenizer", "tokenizer_config.json"),
              {"model_max_length": 77, "bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>",
               "pad_token": "<|endoftext|>", "unk_token": "<|endoftext|>"})
        write(os.path.join(folder, "tokenizer_2", "tokenizer_config.json"),
              {"model_max_length": 512, "eos_token": "</s>", "pad_token": "<pad>", "unk_token": "<unk>"})
    return folder


def write(path, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    text = json.dumps(config, indent=1)
    if not os.path.isfile(path) or open(path, encoding="utf-8").read() != text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)


# ------------------------------------------------------------------------------------- weights in 8 bits

DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
          "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
          "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


def read_state(path, prefix):
    """The tensors under `prefix` in a .safetensors file, with the prefix taken off. They are views of the file
    mapped read-only: safetensors maps it copy-on-write, which Windows charges in full against its memory limit,
    so a 16GB file could crash the process. Nothing here writes into a tensor in place; casts make new ones."""
    import warnings
    import numpy as np
    entries, data = flowfiles.safetensors_entries(path)
    view = np.memmap(path, dtype=np.uint8, mode="r")
    state = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # torch can't mark the view read-only, as said above
        for name, entry in entries.items():
            if not name.startswith(prefix):
                continue
            dtype = DTYPES[entry["dtype"]]
            begin, end = (data + offset for offset in entry["data_offsets"])
            raw = torch.from_numpy(view[begin:end])
            if begin % torch.empty((), dtype=dtype).element_size():
                raw = raw.clone()  # a misaligned view isn't safe for every kernel
            state[name[len(prefix):]] = raw.view(dtype).reshape(entry["shape"])
    return state


NVFP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def unpack_nvfp4(packed, block_scales, global_scale):
    """An NVFP4 weight as float32: two 4-bit codes a byte, high nibble first, one fp8 scale for each 16, kept in
    the tensor cores' 128x4 tile order, and one scale for the tensor."""
    rows, half = packed.shape
    codes = torch.stack(((packed >> 4).long(), (packed & 0x0F).long()), dim=-1).reshape(rows, half * 2)
    values = torch.tensor(NVFP4_LEVELS)[codes]
    blocks = half * 2 // 16
    flat = block_scales.reshape(-1).float()
    if rows % 128 == 0 and blocks % 4 == 0:
        row = torch.arange(rows).reshape(rows, 1).expand(rows, blocks)
        block = torch.arange(blocks).reshape(1, blocks).expand(rows, blocks)
        flat = flat[(row // 128) * (blocks // 4) * 512 + (block // 4) * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + block % 4]
    scales = flat.reshape(rows, blocks)
    return (values.reshape(rows, blocks, 16) * scales.unsqueeze(-1)).reshape(rows, blocks * 16) * float(global_scale.reshape(-1)[0])


def unpack(state):
    """Reads ComfyUI's ways of storing weights: per-tensor tags ("comfy_quant") naming fp8 or NVFP4, the older
    scaled fp8 ("scale_weight"), plain fp8, and bf16, fp16 or fp32. NVFP4 is unpacked to float32, and fp8 stays
    packed. Returns the weights and {module: scale} for the fp8 ones that have a scale."""
    tags = {}
    for name in [n for n in state if n.endswith(".comfy_quant")]:
        tags[name[: -len(".comfy_quant")]] = json.loads(bytes(state.pop(name).reshape(-1).tolist()).decode("utf-8")).get("format", "")
    for module, kind in tags.items():
        if kind not in ("float8_e4m3fn", "float8_e5m2", "nvfp4"):
            raise ValueError(f"{module} is stored as {kind}, which Nano can't read. Use a bf16 or fp8 file.")
    state.pop("scaled_fp8", None)
    weights, scales = {}, {}
    for name, value in state.items():
        module, _, leaf = name.rpartition(".")
        if leaf in ("weight_scale", "weight_scale_2", "scale_input"):
            if leaf == "weight_scale" and tags.get(module, "").startswith("float8"):
                scales[module] = float(value.reshape(-1)[0].float())
            continue  # NVFP4's two scales are read with its weight
        if leaf == "scale_weight":
            scales[module] = float(value.reshape(-1)[0].float())
        elif leaf == "weight" and tags.get(module) == "nvfp4":
            weights[name] = unpack_nvfp4(value, state[module + ".weight_scale"], state[module + ".weight_scale_2"])
        else:
            weights[name] = value
    return weights, scales


def is_fp8(value):
    return value.dtype in (FP8, torch.float8_e5m2)


def eight_bit(state):
    """Big 2-D weights as fp8 with one scale each, everything else in bf16. Returns the state and {module:
    scale}. fp8 weights keep their bytes, and others are scaled into fp8 here."""
    weights, scales = unpack(state)
    out, kept = {}, {}
    for name, value in weights.items():
        module, _, leaf = name.rpartition(".")
        big = value.ndim == 2 and value.numel() >= 2**20 and leaf == "weight"
        if big and is_fp8(value):
            out[name], kept[module] = value.to(FP8), scales.get(module, 1.0)
        elif big and value.is_floating_point() and "embed" not in name and "shared" not in name:
            weight = value.float()
            scale = max(float(weight.abs().max()) / 448, 1e-12)
            out[name], kept[module] = (weight / scale).to(FP8), scale
        elif value.is_floating_point():
            out[name] = (value.float() * scales.get(module, 1.0) if is_fp8(value) else value).to(COMPUTE)
        else:
            out[name] = value
    return out, kept


def full_precision(state):
    """Every weight in bf16, with its scale applied."""
    weights, scales = unpack(state)
    return {name: (value.float() * scales.get(name.rpartition(".")[0], 1.0)).to(COMPUTE) if value.is_floating_point() else value
            for name, value in weights.items()}


def _linear_8(self, x):
    weight = self.weight.to(COMPUTE) * self.fp8_scale
    return torch.nn.functional.linear(x.to(COMPUTE), weight, None if self.bias is None else self.bias.to(COMPUTE))


def _embedding_8(self, x):
    return torch.nn.functional.embedding(x, self.weight.to(COMPUTE) * self.fp8_scale, self.padding_idx)


def cast_per_layer(model, scales):
    """Every fp8 layer casts its weight to bf16 as it computes, times its scale."""
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.dtype != FP8:
            continue
        module.fp8_scale = scales.get(name, 1.0)
        if isinstance(module, torch.nn.Linear):
            module.forward = _linear_8.__get__(module)
        elif isinstance(module, torch.nn.Embedding):
            module.forward = _embedding_8.__get__(module)
        else:
            raise ValueError(f"{name} is fp8 in a layer Nano can't cast ({type(module).__name__}).")
    # T5's feed-forward casts what it computes to its output weight's type, which here would be fp8 and
    # overflow past 448. Its plain form does the same sums without that.
    for module in model.modules():
        if hasattr(module, "wo") and getattr(module.wo, "weight", None) is not None and module.wo.weight.dtype == FP8:
            if hasattr(module, "wi_0"):
                module.forward = (lambda m: lambda h: m.wo(m.dropout(m.act(m.wi_0(h)) * m.wi_1(h))))(module)
            elif hasattr(module, "wi"):
                module.forward = (lambda m: lambda h: m.wo(m.dropout(m.act(m.wi(h)))))(module)
    # Pipelines cast the embeddings to the encoder's dtype, which would read fp8.
    cls = type(model)
    model.__class__ = type(cls.__name__, (cls,), {"dtype": property(lambda self: COMPUTE), "__module__": cls.__module__,
                                                  "__qualname__": cls.__qualname__})
    return model


@contextlib.contextmanager
def no_weights():
    """Builds a model with its parameters on the meta device, so nothing is allocated before the file fills
    it. Buffers stay real, since some are computed when the model is built and aren't in the file."""
    register = torch.nn.Module.register_parameter

    def on_meta(module, name, param):
        register(module, name, param)
        if param is not None:
            module._parameters[name] = torch.nn.Parameter(module._parameters[name].to("meta"), requires_grad=False)

    torch.nn.Module.register_parameter = on_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register


def fill(model, state, label):
    """Loads `state` into a model built by no_weights, and refuses anything that doesn't match."""
    wanted = {n: tuple(t.shape) for n, t in model.state_dict().items()}
    wrong = [n for n, t in state.items() if n in wanted and tuple(t.shape) != wanted[n]]
    if wrong:
        raise ValueError(f"The {label} doesn't match: {wrong[0]} is {tuple(state[wrong[0]].shape)}, not {wanted[wrong[0]]}.")
    model.load_state_dict(state, strict=False, assign=True)
    empty = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if empty:
        raise ValueError(f"The {label} is missing {len(empty)} weights, such as {empty[0]}.")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval()


def text_encoder(cls, config_path, file, prefix, label, eight=True):
    state = read_state(file, prefix)
    state, scales = eight_bit(state) if eight else (full_precision(state), {})
    if "shared.weight" in state and "encoder.embed_tokens.weight" not in state:
        state["encoder.embed_tokens.weight"] = state["shared.weight"]  # T5 ties these
    config = cls.config_class.from_pretrained(config_path)
    with no_weights():
        model = cls(config)
    # transformers 5 dropped CLIP's text_model wrapper, which the files still name.
    own = model.state_dict()
    if not any(k.startswith("text_model.") for k in own) and any(k.startswith("text_model.") for k in state):
        state = {k.removeprefix("text_model."): v for k, v in state.items()}
        scales = {k.removeprefix("text_model."): v for k, v in scales.items()}
    model = fill(model, state, label)
    return cast_per_layer(model, scales) if any(p.dtype == FP8 for p in model.parameters()) else model


def flux2_vae(file, prefix, config_path):
    """Flux.2's autoencoder. diffusers has no single-file loader for it, and ComfyUI's copy keeps the original
    layout, so diffusers' converter for the older autoencoders renames it, the batch-norm terms that Flux.1's
    lacks pass through, and the quantization convolutions move up a level."""
    from diffusers import AutoencoderKLFlux2
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    state = full_precision(read_state(file, prefix))
    fields = {k: v for k, v in AutoencoderKLFlux2.load_config(config_path).items() if not k.startswith("_")}
    if any(".mid.attn_1." in name for name in state):
        converted = convert_ldm_vae_checkpoint(state, fields)
        for name, tensor in state.items():
            for old, new in (("encoder.quant_conv.", "quant_conv."), ("decoder.post_quant_conv.", "post_quant_conv."), ("bn.", "bn.")):
                if name.startswith(old):
                    converted.setdefault(new + name[len(old):], tensor)
        state = converted
    if "bn.running_mean" not in state:
        raise ValueError("The Flux.2 autoencoder is missing its batch-norm terms.")
    with no_weights():
        model = AutoencoderKLFlux2.from_config(fields)  # leaves out what this diffusers version doesn't take
    return fill(model, state, "Flux.2 autoencoder")


def size_for(width, height):
    """Where the flow model paints: the frame's shape, between 512x512 and a megapixel, in multiples of 16."""
    area = min(max(width * height, 512 * 512), 1024 * 1024)
    scale = (area / (width * height)) ** 0.5
    return max(round(width * scale / 16), 1) * 16, max(round(height * scale / 16), 1) * 16


def finish_size(width, height):
    scale = min(1.0, (1024 * 1024 / (width * height)) ** 0.5)
    return max(round(width * scale / 16), 1) * 16, max(round(height * scale / 16), 1) * 16


def signal(sigma):
    """A flow model's noise level as SD's share of signal: its latent, scaled to unit variance, is
    (1 - sigma) of the picture and sigma of noise."""
    sigma = min(max(float(sigma), 0.0), 1.0)
    return (1 - sigma) ** 2 / ((1 - sigma) ** 2 + sigma ** 2)


# ----------------------------------------------------------------------------------------------- the model

class FlowHandoff(HandoffModel):
    """Z-Image, Flux.1 and Flux.2 Klein."""
    size_for = staticmethod(size_for)
    finish_size = staticmethod(finish_size)

    def __init__(self, path, status, device):
        import diffusers
        diffusers.utils.logging.set_verbosity_error()
        super().__init__(device, status)
        self.path = path
        self.kind = model_kind(path)
        names = header(path)
        self.variant = variant(path, self.kind, names)
        self.about = describe(path)
        self.size = self.about.get("size")
        missing = missing_parts(path, self.kind)
        if missing:
            raise RuntimeError(f"{KINDS[self.kind]} needs its {' and '.join(missing)}, which weren't found in the models "
                               f"folder around {os.path.basename(path)}. Put {'them' if len(missing) > 1 else 'it'} there, "
                               "in any subfolder, as ComfyUI keeps them.")
        parts = find_parts(path, self.kind)
        status(f"Loading the {self.label()} handoff model")
        folder = support(self.kind, status)
        started = time.time()
        self.transformer = self.load_transformer(path, folder, names)
        if self.kind == "flux2":
            self.vae = flux2_vae(*parts["vae32"], os.path.join(folder, "vae"))
        else:
            vae_file, vae_prefix = parts["vae16"]
            self.vae = diffusers.AutoencoderKL.from_single_file(vae_file if not vae_prefix else read_state(vae_file, vae_prefix),
                                                                config=os.path.join(folder, "vae"), torch_dtype=COMPUTE).eval()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        if self.kind == "flux2":
            from transformers import Qwen2TokenizerFast, Qwen3Model
            part = "qwen3_8b" if self.size == "9b" else "qwen3_4b"
            self.encoders = [text_encoder(Qwen3Model, os.path.join(folder, f"text_encoder_{self.size}"), *parts[part],
                                          flowfiles.PART_LABELS[part])]
            self.tokenizers = [Qwen2TokenizerFast.from_pretrained(os.path.join(SUPPORT, "zimage", "tokenizer"))]
            scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(FLUX2_SCHEDULER)
            distilled = self.variant != "base"
            self.text_pipe = diffusers.Flux2KleinPipeline(scheduler=scheduler, vae=self.vae, text_encoder=self.encoders[0],
                                                          tokenizer=self.tokenizers[0], transformer=self.transformer,
                                                          is_distilled=distilled)
            # Klein's inpainter would also take the start picture as a reference to edit from, and copy it whatever
            # the noise. So the plain pipeline starts from a picture too, through _picture.
            self.image_pipe = self.text_pipe
        elif self.kind == "zimage":
            from transformers import Qwen2TokenizerFast, Qwen3Model
            self.encoders = [text_encoder(Qwen3Model, os.path.join(folder, "text_encoder"), *parts["qwen3_4b"], "Qwen3-4B text encoder")]
            self.tokenizers = [Qwen2TokenizerFast.from_pretrained(os.path.join(folder, "tokenizer"))]
            scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
                os.path.join(folder, "scheduler"), shift=3.0 if self.variant == "turbo" else 6.0)
            self.text_pipe = diffusers.ZImagePipeline(scheduler=scheduler, vae=self.vae, text_encoder=self.encoders[0],
                                                      tokenizer=self.tokenizers[0], transformer=self.transformer)
            self.image_pipe = diffusers.ZImageImg2ImgPipeline(**self.text_pipe.components)
        else:
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
            self.encoders = [text_encoder(CLIPTextModel, os.path.join(folder, "text_encoder"), *parts["clip_l"], "CLIP-L text encoder", eight=False),
                             text_encoder(T5EncoderModel, os.path.join(folder, "text_encoder_2"), *parts["t5xxl"], "T5-XXL text encoder")]
            self.tokenizers = [CLIPTokenizer.from_pretrained(os.path.join(folder, "tokenizer")),
                               T5TokenizerFast.from_pretrained(os.path.join(folder, "tokenizer_2"))]
            scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_config(FLUX_SCHEDULER["schnell" if self.variant == "schnell" else "dev"])
            self.text_pipe = diffusers.FluxPipeline(scheduler=scheduler, vae=self.vae, text_encoder=self.encoders[0],
                                                    tokenizer=self.tokenizers[0], text_encoder_2=self.encoders[1],
                                                    tokenizer_2=self.tokenizers[1], transformer=self.transformer)
            self.image_pipe = diffusers.FluxImg2ImgPipeline(**self.text_pipe.components)
        for p in (self.text_pipe, self.image_pipe):
            p.set_progress_bar_config(disable=True)
        self.attach_encoders(False)
        self.unet = self.transformer  # what the Handoff code and memory plan call the part that paints
        self.unet_bytes = sum(p.numel() * p.element_size() for p in self.transformer.parameters())
        # Too big to stay on the card beside its working memory at a megapixel, it streams: each block comes to the
        # GPU just before it runs while the next one is copied, and goes back after.
        # Measured against this render's VRAM cap, which keeps room for the desktop.
        usable = (memory.cap_bytes or torch.cuda.get_device_properties(device).total_memory - memory.VRAM_HEADROOM_GB * GB) - 0.4 * GB
        self.streaming = self.unet_bytes + self.vae_bytes() + self.working(1024, 1024) > usable
        if self.streaming:
            # Each block is pinned for the copy as it goes, not kept twice in RAM, pinned and not: that saves the
            # model's size in RAM again, about 5GB for Flux, at no cost in speed.
            self.transformer.enable_group_offload(onload_device=torch.device(device), offload_device=torch.device("cpu"),
                                                  offload_type="block_level", num_blocks_per_group=1, use_stream=True,
                                                  low_cpu_mem_usage=True)
        self.load_seconds = time.time() - started

    def working(self, width, height):
        """Working memory at this size."""
        per_mpx = {"flux": 1.6, "zimage": 1.2, "flux2": 1.4 if self.size == "9b" else 1.0}[self.kind]
        return per_mpx * GB * (width * height) / (1024 * 1024)

    def label(self):
        return self.about["label"]

    def zimage_guidance(self, guidance=None):
        """Z-Image's pipeline counts guidance from 0, one below the usual scale."""
        return max((self.guidance if guidance is None else guidance) - 1, 0.0)

    def load_transformer(self, path, folder, names):
        import diffusers
        cls = {"zimage": diffusers.ZImageTransformer2DModel, "flux": diffusers.FluxTransformer2DModel,
               "flux2": diffusers.Flux2Transformer2DModel}[self.kind]
        config = os.path.join(folder, "transformer")
        if self.kind == "flux":
            channels = {"in_channels": 384, "out_channels": 64} if self.variant == "fill" else {}  # the picture and mask beside the noise
            write(os.path.join(config, "config.json"), {**FLUX_TRANSFORMER, "guidance_embeds": self.variant != "schnell", **channels})
        if self.kind == "flux2":
            config = os.path.join(folder, f"transformer_{self.size}")
            write(os.path.join(config, "config.json"), FLUX2_TRANSFORMER[self.size])
        if path.lower().endswith(".gguf"):
            return cls.from_single_file(path, config=config, torch_dtype=COMPUTE,
                                        quantization_config=diffusers.GGUFQuantizationConfig(compute_dtype=COMPUTE)).eval()
        # A .safetensors model, bf16 or fp8, with ComfyUI's prefix when it holds other parts too. Its names are turned
        # into diffusers' while plain fp8 weights stay packed, then everything big is stored in 8 bits. Scaled fp8 is
        # widened first, since splitting a layer would leave its scale behind.
        from diffusers.loaders import single_file_utils
        prefix = "model.diffusion_model." if any(n.startswith("model.diffusion_model.") for n in names) else ""
        weights, scales = unpack(read_state(path, prefix))
        for module, scale in scales.items():
            if module + ".weight" in weights:
                weights[module + ".weight"] = (weights[module + ".weight"].float() * scale).to(COMPUTE)
        convert = {"zimage": single_file_utils.convert_z_image_transformer_checkpoint_to_diffusers,
                   "flux": single_file_utils.convert_flux_transformer_checkpoint_to_diffusers,
                   "flux2": single_file_utils.convert_flux2_transformer_checkpoint_to_diffusers}[self.kind]
        weights = convert(weights)
        state, scales = eight_bit(weights)
        weights = None
        with no_weights():
            model = cls.from_config(cls.load_config(config))
        model = cast_per_layer(fill(model, state, f"{KINDS[self.kind]} model"), scales)
        gc.collect()
        return model

    def attach_encoders(self, on):
        for p in (self.text_pipe, self.image_pipe):
            if self.kind == "flux":
                p.text_encoder, p.text_encoder_2 = (self.encoders if on else (None, None))
            else:
                p.text_encoder = self.encoders[0] if on else None

    def painters(self):
        """A model that streams is in RAM already."""
        return [] if self.streaming else [self.transformer]

    def to_gpu(self, on):
        self.vae.to(self.device if on else "cpu")
        if not self.streaming:
            self.transformer.to(self.device if on else "cpu")

    @torch.no_grad()
    def encode_text(self, text, negative, guidance, where):
        """One prompt's embeddings, in RAM, as the pipelines take them, for this guidance."""
        if self.kind == "zimage":
            guided = self.zimage_guidance(guidance) > 0
            positive, negatives = self.text_pipe.encode_prompt(prompt=text, device=where, do_classifier_free_guidance=guided,
                                                               negative_prompt=negative or "")
            embeds = {"prompt_embeds": [e.cpu() for e in positive]}
            if guided:
                embeds["negative_prompt_embeds"] = [e.cpu() for e in negatives]
            return embeds
        if self.kind == "flux2":
            positive, _ = self.text_pipe.encode_prompt(prompt=text, device=where, max_sequence_length=512)
            embeds = {"prompt_embeds": positive.cpu()}
            if self.variant == "base" and guidance > 1:  # the distilled builds don't read one
                negatives, _ = self.text_pipe.encode_prompt(prompt=negative or "", device=where, max_sequence_length=512)
                embeds["negative_prompt_embeds"] = negatives.cpu()
            return embeds
        prompt_embeds, pooled, _ = self.text_pipe.encode_prompt(prompt=text, prompt_2=text, device=where,
                                                                 max_sequence_length=256 if self.variant == "schnell" else 512)
        return {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled.cpu()}

    def needs_room(self, width, height, painted=False):
        if not painted:
            width, height = size_for(width, height)
        resident = 0.6 * GB if self.streaming else self.unet_bytes  # streaming holds about two blocks at a time
        return resident + self.vae_bytes() + 0.4 * GB + self.working(width, height)

    def vae_bytes(self):
        return sum(p.numel() * p.element_size() for p in self.vae.parameters())

    def _guidance(self, guidance=None):
        guidance = self.guidance if guidance is None else guidance
        if self.kind == "zimage":
            return {"guidance_scale": self.zimage_guidance(guidance)}
        if self.kind == "flux2":
            return {"guidance_scale": guidance}  # real CFG on Base, and ignored by the distilled builds
        return {"guidance_scale": 0.0 if self.variant == "schnell" else guidance}

    def _picture(self, image, width, height, begin, common):
        """What the picture-to-picture pipelines take to start from `image` at step `begin`. They start at
        int(steps - steps * strength), so half a step less lands on it. Klein has no such pipeline, so its start
        is made here: the picture encoded, noised to the level of step `begin`, and the rest of the path."""
        image = image.convert("RGB").resize((width, height), Image.LANCZOS)
        steps = common["num_inference_steps"]
        if self.kind != "flux2":
            return {"image": image, "strength": (steps - begin - 0.5) / steps}
        import numpy as np
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        pipe = self.text_pipe
        pixels = pipe.image_processor.preprocess(image, height=height, width=width).to(self.device, self.vae.dtype)
        latent = pipe._encode_vae_image(pixels, common["generator"]).float()
        path = np.linspace(1.0, 1 / steps, steps)  # as the pipeline lays it out, then shifts it for the size
        pipe.scheduler.set_timesteps(sigmas=path, device=self.device,
                                     mu=compute_empirical_mu(latent.shape[2] * latent.shape[3], steps))
        sigma = float(pipe.scheduler.sigmas[begin])
        noise = torch.randn(latent.shape, generator=common["generator"], device=self.device, dtype=latent.dtype)
        return {"latents": (1 - sigma) * latent + sigma * noise, "sigmas": list(path[begin:])}

    def _run(self, pipe, run, label, size, common, keep_x0):
        """Runs `pipe` for `run` steps and returns the picture estimate's latent and the noise level after it."""
        scheduler = pipe.scheduler
        step = scheduler.step
        seen = {}

        def keep(model_output, timestep, sample, *a, **k):
            if scheduler.step_index is None:
                scheduler._init_step_index(timestep)
            sigma = scheduler.sigmas[scheduler.step_index]
            seen["x0"] = sample.float() - float(sigma) * model_output.float()
            seen["next"] = float(scheduler.sigmas[scheduler.step_index + 1])
            return step(model_output, timestep, sample, *a, **k)

        def show(values):
            if "x0" in seen:
                self.show(seen["x0"] if keep_x0 else values["latents"], size, common)

        check = self.stepper(label, run, show, last=True)
        scheduler.step = keep
        try:
            out = pipe(callback_on_step_end=check, output_type="latent", **common)
        finally:
            scheduler.__dict__.pop("step", None)
        return (seen.get("x0") if keep_x0 else out.images), seen.get("next", 0.0)

    @torch.no_grad()
    def paint(self, start, repaints, frame, width, height, seed):
        """Like Handoff.paint: the picture estimate where it hands over, and that point's share of signal."""
        sd_w, sd_h = size_for(width, height)
        steps = self.steps
        share = min(max(float(repaints), 0), 1)
        begin = 0 if start is None else min(max(steps - round(share * steps), 0), steps - 1)
        run = round(self.share * (steps - begin))
        if run < 1:
            return None
        shown = start.size if start is not None else (width, height)
        self.next_show = 0
        with self.on_gpu():
            common = dict(num_inference_steps=steps, height=sd_h, width=sd_w, **self._guidance(), **self.embeds_for(frame),
                          generator=self.generator(seed))
            if start is None:
                pipe = self.text_pipe
            else:
                pipe = self.image_pipe
                common.update(self._picture(start, sd_w, sd_h, begin, common))
            x0, sigma = self._run(pipe, run, "Handoff model", shown, common, keep_x0=True)
            self.painter_off()
            picture = self.decode(x0, common)
            self.preview(picture.resize(shown, Image.BILINEAR))
            common = None
        return picture.resize(shown, Image.LANCZOS), signal(sigma)

    @torch.no_grad()
    def finish(self, image, strength, frame, seed):
        width, height = image.size
        sd_w, sd_h = finish_size(width, height)
        run = max(round(self.steps * strength), 1)
        self.next_show = 0
        with self.on_gpu():
            common = dict(num_inference_steps=self.steps, height=sd_h, width=sd_w, **self._guidance(), **self.embeds_for(frame),
                          generator=self.generator(seed))
            common.update(self._picture(image, sd_w, sd_h, max(self.steps - run, 0), common))
            latents, _ = self._run(self.image_pipe, run, "Handoff model finishing", (width, height), common, keep_x0=False)
            self.painter_off()
            picture = self.decode(latents, common)
            common = None
        return picture.resize((width, height), Image.LANCZOS)

    def inpaint_pipe(self):
        """diffusers' inpainting pipeline for this model, built once from the parts already loaded. It paints the
        masked part from `strength` of the way to noise, and at each step puts back the rest of the picture at that
        step's noise level, so the two meet without a seam. Flux Fill sees the picture and mask itself, so it paints
        what fits the area. Klein has its own way, in klein_inpaint()."""
        if getattr(self, "_inpaint_pipe", None) is None:
            import diffusers
            parts = {k: v for k, v in self.text_pipe.components.items() if k in ("scheduler", "vae", "transformer", "tokenizer", "tokenizer_2")}
            if self.kind == "zimage":
                pipe = diffusers.ZImageInpaintPipeline(**parts, text_encoder=None)
            elif self.variant == "fill":
                pipe = diffusers.FluxFillPipeline(**parts, text_encoder=None, text_encoder_2=None)
            else:
                pipe = diffusers.FluxInpaintPipeline(**parts, text_encoder=None, text_encoder_2=None)
            pipe.set_progress_bar_config(disable=True)
            self._inpaint_pipe = pipe
        return self._inpaint_pipe

    def paint_area(self, image, mask, embeds, strength, steps, guidance, seed, check):
        if self.kind == "flux2":
            return self.klein_inpaint(image, mask, embeds, strength, steps, guidance, seed, check)
        out = self.inpaint_pipe()(image=image, mask_image=mask, strength=strength, num_inference_steps=steps, height=image.height,
                                  width=image.width, generator=self.generator(seed), callback_on_step_end=check,
                                  **self._guidance(guidance), **embeds)
        return out.images[0]

    def klein_inpaint(self, image, mask, embeds, strength, steps, guidance, seed, check):
        """Klein's inpainting. diffusers' Klein inpainter always shows the model the picture it starts from as a
        reference, and Klein copies a reference, so what was there comes back. So the plain pipeline paints instead,
        from the picture noised to `strength`, and after every step the rest of the picture goes back in at that
        step's noise level, as the other inpainters do."""
        import numpy as np
        import torch.nn.functional as F
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu
        pipe = self.text_pipe
        generator = self.generator(seed)
        pixels = pipe.image_processor.preprocess(image, height=image.height, width=image.width).to(self.device, self.vae.dtype)
        latent = pipe._encode_vae_image(pixels, generator).float()
        path = np.linspace(1.0, 1 / steps, steps)  # as the pipeline lays it out, then shifts it for the size
        pipe.scheduler.set_timesteps(sigmas=path, device=self.device, mu=compute_empirical_mu(latent.shape[2] * latent.shape[3], steps))
        begin = min(max(steps - max(int(steps * strength), 1), 0), steps - 1)
        sigma = float(pipe.scheduler.sigmas[begin])
        noise = torch.randn(latent.shape, generator=generator, device=self.device, dtype=latent.dtype)
        known, still = pipe._pack_latents(latent), pipe._pack_latents(noise)
        area = imaging.mask_from_pil(mask, self.device)
        area = pipe._pack_latents((F.interpolate(area, latent.shape[2:], mode="area") > 0.5).float())

        def keep_the_rest(p, i, t, values):
            level = float(p.scheduler.sigmas[p.scheduler.step_index])
            painted = values["latents"]
            outside = ((1 - level) * known + level * still).to(painted.dtype)
            values["latents"] = painted * area.to(painted.dtype) + outside * (1 - area).to(painted.dtype)
            return check(p, i, t, values)

        out = pipe(latents=((1 - sigma) * latent + sigma * noise).to(COMPUTE), sigmas=list(path[begin:]), num_inference_steps=steps,
                   height=image.height, width=image.width, generator=generator, callback_on_step_end=keep_the_rest,
                   **self._guidance(guidance), **embeds)
        return out.images[0]

    def decode(self, latent, common, preview=False):
        """Pixels from a latent. Flux packs its latents in 2x2 patches. Klein's are packed too while it paints,
        and normalized by its autoencoder's batch-norm terms, which come off before decoding."""
        pipe = self.text_pipe
        if self.kind == "flux2":
            if latent.ndim == 3:
                h, w = common["height"] // 16, common["width"] // 16
                ids = pipe._prepare_latent_ids(torch.empty(1, 1, h, w)).to(latent.device)
                latent = pipe._unpack_latents_with_ids(latent, ids, h, w)
                mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latent.device, latent.dtype)
                std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latent.device, latent.dtype)
                latent = pipe._unpatchify_latents(latent * std + mean)
            return imaging.to_pil(self.vae.decode(latent.to(COMPUTE)).sample, signed=True)
        if self.kind == "flux":
            latent = pipe._unpack_latents(latent, common["height"], common["width"], pipe.vae_scale_factor)
        latent = latent.to(COMPUTE) / self.vae.config.scaling_factor + self.vae.config.shift_factor
        return imaging.to_pil(self.vae.decode(latent).sample, signed=True)
