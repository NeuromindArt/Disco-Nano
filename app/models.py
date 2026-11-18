"""Model files and their shapes. URLs, hashes and configurations come from Disco Diffusion (MIT)."""
import hashlib
import json
import os
import urllib.request

import atomic
import paths

MAIN = "512x512_diffusion_uncond_finetune_008100"
UPSCALER = "RealESRGAN_x4plus"
PARTNER = "tiny-sd"
PARTNER_URL = "https://huggingface.co/segmind/tiny-sd/resolve/cad0bd7495fa6c4bcca01b19a723dc91627fe84f/"
PARTNER_FILES = {
    "unet/config.json": "55326b8bef1a709254cbb66fdadef156bbe81c011a6be67638a73026be680929",
    "text_encoder/config.json": "e05f77e7bfcd3c155689d90bd3686d36f9ec0d43a4435cf8336198ee3e7c8871",
    "tokenizer/merges.txt": "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a",
    "tokenizer/special_tokens_map.json": "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac",
    "tokenizer/tokenizer_config.json": "19d7b034cb0cc3ce9766c2231373ab8aa8991fc72e2c8f76558bfaae3de0d563",
    "tokenizer/vocab.json": "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349",
    "text_encoder/pytorch_model.bin": "f99d8915149c782551f7d14aae5ca0b6e76a6e94d30e5dc86a349b228fbf92ad",
}
DEPTH = "depth-anything-small"
RIFE = "rife-4.7"
LAMA = "big-lama"
POSE = "keypointrcnn"
PERSON = "maskrcnn"
FACES = "retinaface"
FLUX_FILL = "flux1-fill-dev-Q4_K_S"
# Robust checkers (FARE, Schlarmann et al., MIT), pinned to the revisions Nano was tested with.
ROBUST = {
    "fare4-b16": ("chs20/FARE4-ViT-B-16-laion2B-s34B-b88K", "1e941d54ce261e19ee072f1c3d5af8670aa19c02",
                  "0d432910d3e70ea6fe043662a9fa99472ae87dffef8b1ddd6f5383f96d6dff27", 0.6),
    "fare2-l14": ("chs20/fare2-clip", "6bc37e8efa55395e2f2a413e7b140dc2808032cb",
                  "9fb1f8759c89905809da8ce761a4b90e1c22beaaea7c68f3917647eeb0a230a7", 1.71),
    "fare4-l14": ("chs20/fare4-clip", "8dff9a8a4534c89f6f4b51163569a92d864c4dbb",
                  "eb4b067ab2e2d2b2849a1d477d91baf46c8ceb5faf561e5dd02ab7bf608b5156", 1.71),
}
DEPTH_URL = "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/5426e4f0f36572d16453bbda7a8389317b1bef99/"

FILES = {
    '512x512_diffusion_uncond_finetune_008100': {
        "size_gb": 2.1,
        "sha256": '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648',
        "urls": ['https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt', 'https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt'],
    },
    '256x256_diffusion_uncond': {
        "size_gb": 2.1,
        "sha256": 'a37c32fffd316cd494cf3f35b339936debdc1576dad13fe57c42399a5dbc78b1',
        "urls": ['https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt', 'https://www.dropbox.com/s/9tqnqo930mpnpcn/256x256_diffusion_uncond.pt'],
    },
    'pixelartdiffusion_expanded': {
        "size_gb": 0.42,
        "sha256": 'a73b40556634034bf43b5a716b531b46fb1ab890634d854f5bcbbef56838739a',
        "urls": ['https://huggingface.co/KaliYuga/PADexpanded/resolve/main/PADexpanded.pt'],
    },
    'pixel_art_diffusion_hard_256': {
        "size_gb": 0.42,
        "sha256": 'be4a9de943ec06eef32c65a1008c60ad017723a4d35dc13169c66bb322234161',
        "urls": ['https://huggingface.co/KaliYuga/pixel_art_diffusion_hard_256/resolve/main/pixel_art_diffusion_hard_256.pt'],
    },
    'pixel_art_diffusion_soft_256': {
        "size_gb": 0.42,
        "sha256": 'd321590e46b679bf6def1f1914b47c89e762c76f19ab3e3392c8ca07c791039c',
        "urls": ['https://huggingface.co/KaliYuga/pixel_art_diffusion_soft_256/resolve/main/pixel_art_diffusion_soft_256.pt'],
    },
    'pixelartdiffusion4k': {
        "size_gb": 0.42,
        "sha256": 'a1ba4f13f6dabb72b1064f15d8ae504d98d6192ad343572cc416deda7cccac30',
        "urls": ['https://huggingface.co/KaliYuga/pixelartdiffusion4k/resolve/main/pixelartdiffusion4k.pt'],
    },
    'watercolordiffusion_2': {
        "size_gb": 0.42,
        "sha256": '49c281b6092c61c49b0f1f8da93af9b94be7e0c20c71e662e2aa26fee0e4b1a9',
        "urls": ['https://huggingface.co/KaliYuga/watercolordiffusion_2/resolve/main/watercolordiffusion_2.pt'],
    },
    'watercolordiffusion': {
        "size_gb": 0.42,
        "sha256": 'a3e6522f0c8f278f90788298d66383b11ac763dd5e0d62f8252c962c23950bd6',
        "urls": ['https://huggingface.co/KaliYuga/watercolordiffusion/resolve/main/watercolordiffusion.pt'],
    },
    'PulpSciFiDiffusion': {
        "size_gb": 0.42,
        "sha256": 'b79e62613b9f50b8a3173e5f61f0320c7dbb16efad42a92ec94d014f6e17337f',
        "urls": ['https://huggingface.co/KaliYuga/PulpSciFiDiffusion/resolve/main/PulpSciFiDiffusion.pt'],
    },
    'secondary': {
        "size_gb": 0.055,
        "sha256": '983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a',
        "urls": ['https://huggingface.co/spaces/huggi/secondary_model_imagenet_2.pth/resolve/main/secondary_model_imagenet_2.pth', 'https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth', 'https://ipfs.pollinations.ai/ipfs/bafybeibaawhhk7fhyhvmm7x24zwwkeuocuizbqbcg5nqx64jq42j75rdiy/secondary_model_imagenet_2.pth'],
    },
    UPSCALER: {
        "size_gb": 0.064,
        "sha256": '4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1',
        "urls": ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'],
        "folder": "upscale_models",
    },
    PARTNER: {
        "size_gb": 0.83,
        "sha256": "c9b63e57cb1479c946c73bf5484bc1a823dbe565158ea0ee40015a18844d9f29",
        "urls": [PARTNER_URL + "unet/diffusion_pytorch_model.bin"],
        "folder": "tiny-sd/unet",
        "needs": [f"{PARTNER}/{path}" for path in PARTNER_FILES],  # this file comes last, so having it means all are here
    },
    DEPTH: {
        "size_gb": 0.1,
        "sha256": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
        "urls": [DEPTH_URL + "model.safetensors"],
        "folder": DEPTH,
        "needs": [f"{DEPTH}/config.json"],
    },
    POSE: {
        "size_gb": 0.24,
        "sha256": "fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926",
        "urls": ["https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth"],
        "folder": "pose",
    },
    PERSON: {
        "size_gb": 0.19,
        "sha256": "73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e",
        "urls": ["https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"],
        "folder": "person",
    },
    FACES: {
        "size_gb": 0.11,
        "sha256": "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
        "urls": ["https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"],
        "folder": "faces",
    },
    RIFE: {
        "size_gb": 0.021,
        "sha256": "fcf3492b10f17fb035156ea4177ed87b1f517eae54fe4500e878f5d186043d5e",
        "urls": ["https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v4.7.pkl"],
        "folder": "rife",
    },
    LAMA: {  # LaMa's big-lama as TorchScript, as Lama Cleaner and ComfyUI's inpaint nodes use it
        "size_gb": 0.21,
        "sha256": "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9",
        "urls": ["https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"],
        "folder": "lama",
    },
    FLUX_FILL: {
        "size_gb": 6.34,
        "sha256": "3782c22eda0622d50eb8683d2aa664e8534a8c1e29ce9e91df491cc400944dc8",
        "urls": ["https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF/resolve/78b83f1da140a4dcd6466580516544e1f6effe3e/flux1-fill-dev-Q4_K_S.gguf"],
        "folder": "checkpoints",
    },
}
# The Robust checker setting's choices, and open_clip's name for each network.
ROBUST_CHOICES = {"FARE4 B/16": ("fare4-b16", "ViT-B-16"), "FARE2 L/14": ("fare2-l14", "ViT-L-14-quickgelu"),
                  "FARE4 L/14": ("fare4-l14", "ViT-L-14-quickgelu")}

FILES.update({name: {"size_gb": size, "sha256": sha256, "folder": f"robust/{name}",
                     "urls": [f"https://huggingface.co/{repo}/resolve/{revision}/open_clip_pytorch_model.bin"]}
              for name, (repo, revision, sha256, size) in ROBUST.items()})

DIFFUSION = [name for name in FILES if name not in ("secondary", PARTNER, DEPTH, POSE, PERSON, FACES, UPSCALER, RIFE, LAMA, FLUX_FILL, *ROBUST)]

# Small helpers that download when first needed and aren't listed in the app.
HELPERS = {
    "taesd_encoder": {"sha256": "15bc6128f0ac51c673d3427216082ebfa62402ffc89763af74edfe259b32d49d", "folder": "taesd",
                      "urls": ["https://github.com/madebyollin/taesd/raw/main/taesd_encoder.pth"]},
    "taesd_decoder": {"sha256": "4ded74b856789e00948e8b8e920de92b1405756d2ee6f24842a90ffd9281cf68", "folder": "taesd",
                      "urls": ["https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth"]},
    **{f"{PARTNER}/{path}": {"sha256": sha256, "urls": [PARTNER_URL + path], "folder": os.path.dirname(f"{PARTNER}/{path}")}
       for path, sha256 in PARTNER_FILES.items()},
    f"{DEPTH}/config.json": {"sha256": "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75", "folder": DEPTH,
                             "urls": [DEPTH_URL + "config.json"]},
}
# Configs and tokenizers for Z-Image and Flux handoff models, pinned. Flux's own repos are gated, so its configs are
# written by flow.py, and its tokenizers come from the open repos they were made from: CLIP's vocabulary from LAION
# (MIT), and T5's from t5-base (Apache 2.0), whose spiece.model is the same file as T5 v1.1-XXL's.
ZIMAGE_URL = "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/resolve/f332072aa78be7aecdf3ee76d5c247082da564a6/"
FLOW_FILES = {
    "zimage": {
        ZIMAGE_URL + "transformer/config.json": ("transformer", "06d49d1ae533f825c4704cc616fc8bc7826e9d449dde292cc7a95163c69af24a"),
        ZIMAGE_URL + "text_encoder/config.json": ("text_encoder", "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"),
        ZIMAGE_URL + "scheduler/scheduler_config.json": ("scheduler", "3b979ab0956e4f5e8d02ec409ac6a4ece1555191d15bd10788fdc85fea5d13fc"),
        ZIMAGE_URL + "tokenizer/tokenizer.json": ("tokenizer", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
        ZIMAGE_URL + "tokenizer/tokenizer_config.json": ("tokenizer", "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
        ZIMAGE_URL + "tokenizer/vocab.json": ("tokenizer", "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        ZIMAGE_URL + "tokenizer/merges.txt": ("tokenizer", "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    },
    "flux": {
        "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/743c27bd53dfe508a0ade0f50698f99b39d03bec/vocab.json":
            ("tokenizer", "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"),
        "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/743c27bd53dfe508a0ade0f50698f99b39d03bec/merges.txt":
            ("tokenizer", "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
        "https://huggingface.co/google-t5/t5-base/resolve/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/tokenizer.json":
            ("tokenizer_2", "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066"),
    },
}
FLOW_SUPPORT = {kind: [f"handoff-support/{kind}/{folder}/{url.rsplit('/', 1)[-1]}" for url, (folder, _) in files.items()]
                for kind, files in FLOW_FILES.items()}
FLOW_SUPPORT["flux2"] = [name for name in FLOW_SUPPORT["zimage"] if "/tokenizer/" in name]  # the same Qwen3 tokenizer
HELPERS.update({f"handoff-support/{kind}/{folder}/{url.rsplit('/', 1)[-1]}":
                {"sha256": sha256, "urls": [url], "folder": f"handoff-support/{kind}/{folder}"}
                for kind, files in FLOW_FILES.items() for url, (folder, sha256) in files.items()})
ALL_FILES = {**FILES, **HELPERS}

# Plain names for the model list. The official names are the file names.
LABELS = {
    MAIN: "Main 512",
    PARTNER: "Tiny SD",
    DEPTH: "Depth Anything V2 Small",
    RIFE: "RIFE 4.7",
    LAMA: "LaMa",
    POSE: "Keypoint R-CNN",
    PERSON: "Mask R-CNN",
    FACES: "RetinaFace",
    FLUX_FILL: "Flux Fill",
    "fare4-b16": "Robust checker FARE4 B/16",
    "fare2-l14": "Robust checker FARE2 L/14",
    "fare4-l14": "Robust checker FARE4 L/14",
    "256x256_diffusion_uncond": "Classic 256",
    "pixelartdiffusion_expanded": "Pixel art, expanded",
    "pixel_art_diffusion_hard_256": "Pixel art, hard",
    "pixel_art_diffusion_soft_256": "Pixel art, soft",
    "pixelartdiffusion4k": "Pixel art 4K",
    "watercolordiffusion_2": "Watercolor 2",
    "watercolordiffusion": "Watercolor",
    "PulpSciFiDiffusion": "Pulp sci-fi",
    "secondary": "Secondary model",
    UPSCALER: "Real-ESRGAN x4plus",
}

ABOUT = {
    MAIN: "The standard model and the best all-rounder.",
    PARTNER: "Tiny SD, the small text-to-image model behind Prompt graft, Live ghost, Two painters and Borrowed attention. It never paints the final image.",
    DEPTH: "Depth Anything V2 Small, which judges how near each part of the picture looks. Depth target, Depth control, Face shape and 3D animation need it.",
    "fare4-b16": "A robust checker for the Experiments card. Whole scenes and steady subjects, at almost no extra cost. By Schlarmann et al.",
    "fare2-l14": "A larger robust checker. Slower, with bolder scenes. By Schlarmann et al.",
    "fare4-l14": "The most robust of the large checkers. Slower, and pulls hardest toward shapes. By Schlarmann et al.",
    FACES: "RetinaFace ResNet-50 by Deng et al., which finds faces for Face control and Face detail. Downloads by itself the first time either is used.",
    PERSON: "Mask R-CNN from torchvision, which cuts out people's silhouettes for Person control. Downloads by itself the first time Person control is used.",
    POSE: "Keypoint R-CNN from torchvision, which finds people's joints for Pose control. Downloads by itself the first time Pose control is used.",
    RIFE: "RIFE 4.7 by Zhewei Huang, which paints smooth in-between frames for animations. Without it, optical flow is used.",
    LAMA: "LaMa by Suvorov et al., which fills what Erase removes during a render. Downloads by itself the first time Erase is used.",
    FLUX_FILL: "Flux.1 Fill dev by Black Forest Labs, made for inpainting, for Fix's Inpaint. It uses the Flux text encoders and VAE in your model folders. Non-commercial license.",
    "256x256_diffusion_uncond": "An older general model. Slower and heavier than the main model at 512×512.",
    "pixelartdiffusion_expanded": "Pixel art with strong color.",
    "pixel_art_diffusion_hard_256": "Muted pixel art.",
    "pixel_art_diffusion_soft_256": "Softer pixel art.",
    "pixelartdiffusion4k": "Crisp pixel art with strong color.",
    "watercolordiffusion_2": "Bold watercolor.",
    "watercolordiffusion": "Quiet, misty watercolor.",
    "PulpSciFiDiffusion": "The look of old pulp magazine covers.",
    "secondary": "A small helper that makes large renders faster, at lower quality.",
    UPSCALER: "A general 4× upscaler by Xintao Wang.",
}

PIXEL_ART = {"pixelartdiffusion_expanded", "pixel_art_diffusion_hard_256", "pixel_art_diffusion_soft_256",
             "pixelartdiffusion4k", "PulpSciFiDiffusion"}
WATERCOLOR = {"watercolordiffusion", "watercolordiffusion_2"}

# Style models work best with their own cutout schedules.
CUT_SCHEDULES = {
    "pixel_art": {
        "cut_overview": "[15]*100+[15]*100+[12]*100+[12]*100+[6]*100+[4]*100+[2]*200+[0]*200",
        "cut_innercut": "[1]*100+[1]*100+[4]*100+[4]*100+[8]*100+[8]*100+[10]*200+[10]*200",
        "cut_ic_pow": "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300",
        "cut_icgray_p": "[0.87]*100+[0.78]*50+[0.73]*50+[0.64]*60+[0.56]*40+[0.50]*50+[0.33]*100+[0.19]*150+[0]*400",
    },
    "watercolor": {
        "cut_overview": "[14]*200+[12]*200+[4]*400+[0]*200",
        "cut_innercut": "[2]*200+[4]*200+[12]*400+[12]*200",
        "cut_ic_pow": "[12]*300+[12]*100+[12]*50+[12]*50+[10]*100+[10]*100+[10]*300",
        "cut_icgray_p": "[0.7]*100+[0.6]*100+[0.45]*100+[0.3]*100+[0]*600",
    },
}


def filename(name):
    return ALL_FILES[name]["urls"][0].split("?")[0].rsplit("/", 1)[-1]


def own_path(name):
    return os.path.join(paths.MODELS, ALL_FILES[name].get("folder", ""), filename(name))


def path_of(name):
    """Nano's own copy, or the same file found in another app's folder, which is used where it is."""
    own = own_path(name)
    if os.path.exists(own):
        return own
    there = found_copies().get(name)
    return there if there and os.path.exists(there) else own


# Models found in other apps' folders, by name, and where each is. A copy that is moved or deleted later is
# simply downloaded again.
FOUND = os.path.join(paths.MODELS, "found.json")
_found = {"key": None, "copies": {}}


def found_copies():
    try:
        key = os.path.getmtime(FOUND)
    except OSError:
        return {}
    if key != _found["key"]:
        try:
            with open(FOUND, encoding="utf-8") as f:
                _found["copies"] = json.load(f)
        except (OSError, ValueError):
            _found["copies"] = {}
        _found["key"] = key
    return _found["copies"]


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_copies(roots, report=lambda message: None):
    """Looks under the roots for models Nano doesn't have yet, and keeps those that are the same file: the same name,
    and the SHA-256 pinned here. Returns the names it found."""
    import finder
    wanted = {}
    for name, entry in ALL_FILES.items():
        if entry.get("sha256") and not os.path.exists(path_of(name)):
            wanted.setdefault(filename(name).lower(), []).append(name)
    if not wanted:
        return []
    copies, new = dict(found_copies()), []
    for path in finder.files_named(roots, set(wanted)):
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        for name in wanted[os.path.basename(path).lower()]:
            expected = ALL_FILES[name].get("size_gb")
            # The size rules out most files that only share a name, such as model.safetensors, before the slow part.
            if name in new or (expected is None and size > 50 * 2**20) or \
                    (expected is not None and abs(size / 1e9 - expected) > max(0.03, expected * 0.15)):
                continue
            report(f"Checking {os.path.basename(path)}")
            try:
                same = sha256_of(path) == ALL_FILES[name]["sha256"]
            except OSError:
                continue
            if same:
                copies[name] = path
                new.append(name)
    if new:
        os.makedirs(paths.MODELS, exist_ok=True)
        atomic.write_json(FOUND, copies, indent=1)
    return new


def ensure(name, report):
    """Returns a model file's path, downloading and checking it the first time."""
    for part in ALL_FILES[name].get("needs", ()):
        ensure(part, report)
    path = path_of(name)
    if os.path.exists(path):
        return path
    path = own_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    label = os.path.basename(path)
    for url in ALL_FILES[name]["urls"]:
        # A file of its own, so the server and a render fetching the same model at once don't write into one file.
        part = atomic.temp_of(path)
        try:
            download(url, part, lambda percent: report(f"Downloading {label} ({percent}%)"))
            report(f"Checking {label}")
            same = sha256_of(part) == ALL_FILES[name]["sha256"]
        except OSError:  # no connection, a stalled one, or a full disk: the next URL, if there is one
            same = False
        if same and (atomic.replace(part, path) or os.path.exists(path)):  # or the other one got there first
            return path
        try:
            os.remove(part)
        except OSError:
            pass
    raise RuntimeError(f"Could not download {name}.")


STALLED = 60  # seconds without a byte before a download gives up


def download(url, path, progress):
    """Fetches `url` into `path`, calling progress(percent) as it goes. Raises OSError when it fails or stalls."""
    with urllib.request.urlopen(url, timeout=STALLED) as response, open(path, "wb") as f:
        total, done, shown = int(response.headers.get("Content-Length") or 0), 0, None
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            percent = min(done * 100 // total, 100) if total else 0
            if percent != shown:
                shown = percent
                progress(percent)
    if total and done < total:
        raise OSError(f"The download of {url} ended early.")


def config(name):
    """Settings for guided-diffusion's create_model_and_diffusion."""
    common = {"class_cond": False, "rescale_timesteps": True, "learn_sigma": True, "noise_schedule": "linear",
              "num_res_blocks": 2, "use_scale_shift_norm": True, "use_fp16": True}
    if name == "512x512_diffusion_uncond_finetune_008100" or name == "256x256_diffusion_uncond":
        return {**common, "attention_resolutions": "32, 16, 8", "image_size": 512 if name.startswith("512") else 256,
                "num_channels": 256, "num_head_channels": 64, "resblock_updown": True}
    return {**common, "attention_resolutions": "16", "image_size": 256, "num_channels": 128, "num_heads": 1,
            "use_scale_shift_norm": False}


def cut_style(name):
    return "pixel_art" if name in PIXEL_ART else "watercolor" if name in WATERCOLOR else None
