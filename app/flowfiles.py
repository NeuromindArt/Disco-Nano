"""Z-Image, Flux.1 and Flux.2 Klein model files for Handoff: what kind a file is, and where its other parts are. It reads only
file headers and a few values, without torch, so the server can list models quickly. flow.py loads them."""
import json
import os
import struct

KINDS = {"zimage": "Z-Image", "flux": "Flux", "flux2": "Flux.2 Klein"}


def first_values(path, name, count):
    """The first `count` values of one tensor in a .safetensors file, as floats."""
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(8), "little")
        entry = json.loads(f.read(length))[name]
        size = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1}[entry["dtype"]]
        f.seek(8 + length + entry["data_offsets"][0])
        raw = f.read(size * count)
    if entry["dtype"] == "F32":
        return list(struct.unpack(f"<{count}f", raw))
    if entry["dtype"] == "F16":
        return list(struct.unpack(f"<{count}e", raw))
    if entry["dtype"] == "BF16":  # the top half of a float32
        return [struct.unpack("<f", b"\0\0" + raw[i:i + 2])[0] for i in range(0, 2 * count, 2)]
    values = []
    for b in raw:  # fp8 e4m3: a sign, four exponent bits and three of mantissa
        exponent, mantissa = (b >> 3) & 15, b & 7
        value = mantissa / 8 * 2**-6 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
        values.append(-value if b & 128 else value)
    return values

def header(path):
    """{tensor name: shape} from a .safetensors or .gguf file, without reading the weights. GGUF lists shapes
    reversed, so they're turned around to match."""
    if path.lower().endswith(".gguf"):
        import gguf
        reader = gguf.GGUFReader(path)
        return {t.name: list(reversed([int(n) for n in t.shape])) for t in reader.tensors}
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(8), "little")
        found = json.loads(f.read(length)) if 0 < length < 200 * 2**20 else {}
    found.pop("__metadata__", None)
    return {name: entry["shape"] for name, entry in found.items()}


def find_prefix(names, suffix, shape=None):
    """The prefix before `suffix` in the first name that ends with it and has `shape`, or None."""
    for name, size in names.items():
        if name.endswith(suffix) and (shape is None or list(size) == list(shape)):
            return name[: -len(suffix)]
    return None


def model_kind(path):
    """"zimage", "flux" or "flux2" for a model Handoff can use, or None. Flux Fill, Canny and Depth take more
    input channels, so those are left out, and of Flux.2 only Klein 4B and 9B fit."""
    try:
        names = header(path)
    except Exception:
        return None
    if find_prefix(names, "cap_embedder.1.weight") is not None and find_prefix(names, "noise_refiner.0.attention.qkv.weight") is not None:
        return "zimage"
    if klein_size(names):
        return "flux2"
    prefix = find_prefix(names, "double_blocks.0.img_attn.qkv.weight")
    # Chroma is built from Flux but drops its CLIP input, vector_in, and steers itself another way.
    if prefix is not None and find_prefix(names, "single_blocks.37.linear1.weight") is not None \
            and prefix + "vector_in.in_layer.weight" in names:
        size = names.get(prefix + "img_in.weight")
        if size and sorted(size) == [64, 3072]:
            return "flux"
    return None


def klein_size(names):
    """"4b" or "9b" for a Flux.2 Klein model, by its width and depth, or None. Flux.2 dev is far bigger than
    an 8GB card can take beside Disco, so it isn't one of them."""
    prefix = find_prefix(names, "double_stream_modulation_img.lin.weight")
    if prefix is None:
        return None
    size = names.get(prefix + "img_in.weight")
    blocks = len({n.split(".")[1] for n in names if n.startswith(prefix + "double_blocks.")})
    if size and sorted(size) == [128, 3072] and blocks == 5:
        return "4b"
    if size and sorted(size) == [128, 4096] and blocks == 8:
        return "9b"
    return None


# Steps and CFG each model was published with, set when it is first chosen. CFG is on the usual scale, where 1 is
# off: Flux dev's is its built-in guidance, and schnell and the distilled Kleins have none.
DEFAULTS = {("zimage", "turbo"): {"handoff_steps": 8, "handoff_guidance": 1},
            ("zimage", "base"): {"handoff_steps": 25, "handoff_guidance": 4},
            ("flux", "dev"): {"handoff_steps": 20, "handoff_guidance": 3.5},
            ("flux", "schnell"): {"handoff_steps": 4, "handoff_guidance": 1},
            ("flux2", "klein"): {"handoff_steps": 4, "handoff_guidance": 1},
            ("flux2", "base"): {"handoff_steps": 20, "handoff_guidance": 5}}


def describe(path):
    """{kind, variant, label, defaults, and for Klein its size} for the Handoff list, or None for a model it
    can't use."""
    kind = model_kind(path)
    if not kind:
        return None
    names = header(path)
    which = variant(path, kind, names)
    about = {"kind": kind, "variant": which, "defaults": DEFAULTS[(kind, which)]}
    if kind == "zimage":
        about["label"] = "Z-Image Turbo" if which == "turbo" else "Z-Image"
    elif kind == "flux":
        about["label"] = f"Flux {which}"
    else:
        about["size"] = klein_size(names)
        about["label"] = f"Flux.2 Klein{' Base' if which == 'base' else ''} {about['size'].upper()}"
    return about


def variant(path, kind, names):
    """Z-Image Turbo or Base, and Klein or Klein Base, from the file's or its folder's name, since the weights
    look alike. Flux dev has a guidance embedding, which schnell lacks."""
    # Only the file's own name and its folder's: folders further up are often sorted by family, such as ZImageBase.
    near = os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)).lower()
    if kind == "zimage":
        return "turbo" if "turbo" in near else "base"
    if kind == "flux2":
        return "base" if "base" in near else "klein"
    return "dev" if find_prefix(names, "guidance_in.in_layer.weight") is not None else "schnell"


# What each part looks like inside a file: tensor names' endings with their shapes, all under one prefix. Some
# other models share every shape, such as SD 3's autoencoder and Qwen3-VL's text half, so the first few values
# of a norm tell those apart, and a file with vision weights is another model.
PARTS = {
    "qwen3_4b": {"keys": {"embed_tokens.weight": [151936, 2560]}, "values": ("norm.weight", [0.0055, 2.2031, 3.6562, 1.4297])},
    "qwen3_8b": {"keys": {"embed_tokens.weight": [151936, 4096]}, "values": ("norm.weight", [2.125, 1.9219, 2.0469, 2.0312])},
    "vae32": {"keys": {"decoder.conv_in.weight": [512, 32, 3, 3], "bn.running_mean": [128]}},
    "t5xxl": {"keys": {"shared.weight": [32128, 4096], "encoder.block.23.layer.1.DenseReluDense.wo.weight": [4096, 10240]}},
    "clip_l": {"keys": {"text_model.embeddings.token_embedding.weight": [49408, 768],
                        "text_model.embeddings.position_embedding.weight": [77, 768]}},
    "vae16": {"keys": {"decoder.conv_in.weight": [512, 16, 3, 3]}, "values": ("decoder.conv_in.bias", [-0.0569, 0.0496, -0.0628, -0.1179])},
}


def part_prefix(path, names, part):
    """The prefix `part` sits under in the file, or None when it isn't there."""
    want = PARTS[part]
    first, shape = next(iter(want["keys"].items()))
    prefix = find_prefix(names, first, shape)
    if prefix is None or any(list(names.get(prefix + k, [])) != v for k, v in want["keys"].items()):
        return None
    if any(".visual." in n or n.startswith(prefix + "visual.") for n in names):
        return None
    if "values" in want:
        name, expected = want["values"]
        try:
            found = first_values(path, prefix + name, len(expected))
        except (OSError, KeyError, ValueError):
            return None
        if any(abs(a - b) > 0.04 for a, b in zip(found, expected)):
            return None
    return prefix


def needs(kind, names):
    """The parts `kind` needs. Klein 4B reads its prompt with Qwen3-4B, as Z-Image does, and Klein 9B with
    Qwen3-8B, and both share Flux.2's own autoencoder."""
    if kind == "flux2":
        return ["qwen3_8b" if klein_size(names) == "9b" else "qwen3_4b", "vae32"]
    return {"zimage": ["qwen3_4b", "vae16"], "flux": ["t5xxl", "clip_l", "vae16"]}[kind]


NAME_HINTS = {"qwen3_4b": ("qwen",), "qwen3_8b": ("qwen",), "t5xxl": ("t5",),
              "clip_l": ("clip_l", "clip-l", "clip_vit_l", "clip"), "vae16": ("ae.", "vae", "flux"),
              "vae32": ("vae", "flux2", "ae.")}
PART_LABELS = {"qwen3_4b": "Qwen3-4B text encoder", "qwen3_8b": "Qwen3-8B text encoder", "t5xxl": "T5-XXL text encoder",
               "clip_l": "CLIP-L text encoder", "vae16": "Flux autoencoder (ae.safetensors)",
               "vae32": "Flux.2 autoencoder (flux2-vae.safetensors)"}


def search_roots(path):
    """Where a model's other parts may be: the nearest folder called models above it, as in ComfyUI and the
    web UIs, or else the folder two above it."""
    folder = os.path.dirname(os.path.abspath(path))
    here = folder
    for _ in range(6):
        if os.path.basename(here).lower() == "models":
            return [here]
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return [os.path.dirname(os.path.dirname(folder)) or folder]


_found = {}


def find_parts(path, kind):
    """{part: (file, prefix)} for what `kind` needs, from inside the model's own file first, then from the
    files around it. Missing parts are left out. A file whose name doesn't hint at the part isn't opened,
    which keeps the search to a few headers."""
    key = (path, kind)
    if key in _found and all(os.path.isfile(f) for f, _ in _found[key].values()):
        return _found[key]
    names = header(path)
    wanted = needs(kind, names)
    found = {}
    for part in wanted:
        prefix = None if path.lower().endswith(".gguf") else part_prefix(path, names, part)
        if prefix is not None:
            found[part] = (path, prefix)
    candidates = []
    for root in search_roots(path):
        for folder, subfolders, files in os.walk(root):
            depth = folder.count(os.sep) - root.count(os.sep)
            subfolders[:] = [s for s in subfolders if not s.startswith(".")] if depth < 5 else []
            candidates += [os.path.join(folder, f) for f in files if f.lower().endswith(".safetensors")]
    for part in wanted:
        if part in found:
            continue
        hinted = [c for c in candidates if any(h in os.path.basename(c).lower() for h in NAME_HINTS[part])]
        # Smaller first: an fp8 encoder over its bf16 twin, which loads faster and ends up the same in memory.
        for candidate in sorted(hinted, key=lambda c: os.path.getsize(c)):
            try:
                prefix = part_prefix(candidate, header(candidate), part)
            except Exception:
                continue
            if prefix is not None:
                found[part] = (candidate, prefix)
                break
    _found[key] = found
    return found


def missing_parts(path, kind):
    """Plain names of the parts `kind` needs that weren't found, for a message."""
    have = find_parts(path, kind)
    return [PART_LABELS[p] for p in needs(kind, header(path)) if p not in have]
