"""Model files and their shapes. URLs, hashes and configurations come from Disco Diffusion (MIT)."""
import hashlib
import os
import urllib.request

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
POSE = "keypointrcnn"
PERSON = "maskrcnn"
FACES = "retinaface"
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
}
# The Robust checker setting's choices, and open_clip's name for each network.
ROBUST_CHOICES = {"FARE4 B/16": ("fare4-b16", "ViT-B-16"), "FARE2 L/14": ("fare2-l14", "ViT-L-14-quickgelu"),
                  "FARE4 L/14": ("fare4-l14", "ViT-L-14-quickgelu")}

FILES.update({name: {"size_gb": size, "sha256": sha256, "folder": f"robust/{name}",
                     "urls": [f"https://huggingface.co/{repo}/resolve/{revision}/open_clip_pytorch_model.bin"]}
              for name, (repo, revision, sha256, size) in ROBUST.items()})

DIFFUSION = [name for name in FILES if name not in ("secondary", PARTNER, DEPTH, POSE, PERSON, FACES, UPSCALER, RIFE, *ROBUST)]

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
ALL_FILES = {**FILES, **HELPERS}

# Plain names for the model list. The official names are the file names.
LABELS = {
    MAIN: "Main 512",
    PARTNER: "Tiny SD",
    DEPTH: "Depth Anything V2 Small",
    RIFE: "RIFE 4.7",
    POSE: "Keypoint R-CNN",
    PERSON: "Mask R-CNN",
    FACES: "RetinaFace",
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
    PARTNER: "Tiny SD, the small text model behind Prompt graft, Live ghost, Two painters and Borrowed attention. It never paints the final image.",
    DEPTH: "Depth Anything V2 Small, which judges how near each part of the picture looks. Depth target needs it.",
    "fare4-b16": "A robust checker for the Experiments card. Whole scenes and steady subjects, at almost no extra cost. By Schlarmann et al.",
    "fare2-l14": "A larger robust checker. Slower, with bolder scenes. By Schlarmann et al.",
    "fare4-l14": "The most robust of the large checkers. Slower, and pulls hardest toward shapes. By Schlarmann et al.",
    FACES: "RetinaFace ResNet-50 by Deng et al., which finds faces for Face control and Face detail. Downloads by itself the first time either is used.",
    PERSON: "Mask R-CNN from torchvision, which cuts out people's silhouettes for Person control. Downloads by itself the first time Person control is used.",
    POSE: "Keypoint R-CNN from torchvision, which finds people's joints for Pose control. Downloads by itself the first time Pose control is used.",
    RIFE: "RIFE 4.7 by Zhewei Huang, which paints smooth in-between frames for animations. Without it, optical flow is used.",
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


def path_of(name):
    return os.path.join(paths.MODELS, ALL_FILES[name].get("folder", ""), filename(name))


def ensure(name, report):
    """Returns a model file's path, downloading and checking it the first time."""
    for part in ALL_FILES[name].get("needs", ()):
        ensure(part, report)
    path = path_of(name)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part, label = path + ".part", os.path.basename(path)
    for url in ALL_FILES[name]["urls"]:
        shown = [None]

        def progress(blocks, block_size, total):
            percent = min(blocks * block_size * 100 // total, 100) if total > 0 else 0
            if percent != shown[0]:
                shown[0] = percent
                report(f"Downloading {label} ({percent}%)")

        try:
            urllib.request.urlretrieve(url, part, progress)
        except OSError:
            continue
        report(f"Checking {label}")
        digest = hashlib.sha256()
        with open(part, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                digest.update(chunk)
        if digest.hexdigest() == ALL_FILES[name]["sha256"]:
            os.replace(part, path)
            return path
        os.remove(part)
    raise RuntimeError(f"Could not download {name}.")


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
