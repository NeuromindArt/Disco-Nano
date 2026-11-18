# Credits

Disco Diffusion Nano exists because of the people below. It is an independent project and is not
affiliated with or endorsed by any of them.

## Disco Diffusion

`app/disco.py`, with its parts in `app/guidance.py`, `app/secondary.py`, `app/loaders.py` and `app/prompts.py`, and the
model list in `app/models.py` are derived from
[Disco Diffusion](https://github.com/alembics/disco-diffusion), which is MIT licensed. Its license is
in [LICENSES/disco-diffusion.txt](LICENSES/disco-diffusion.txt). These credits follow its notebook.

- **Katherine Crowson** ([crowsonkb](https://github.com/crowsonkb)): the original notebook, CLIP guided diffusion, and the fine-tuned 512x512 model
- **Daniel Russell** ([russelldc](https://github.com/russelldc)): parameters for short runs and more robust augmentations
- **Dango233**: quality improvements and the Dango cutout method
- **nshepperd**: quality improvements and horizontal and vertical symmetry
- **Vark**: loading several CLIP models at once
- **Somnai**: Disco Diffusion's animation techniques, quality-of-life improvements and many others
- **Adam Letts** (gandamu): 3D animation with Somnai, the creation of disco.py and its maintenance
- **Chigozie Nri** ([chigozienri](https://github.com/chigozienri)): zoom, pan, rotation and keyframes
- **Chris Allen** (zippy731): Turbo
- **HostsServer**: local running, Windows support and dependency installation
- **Tom Mason**: VR mode
- **huemin**: symmetry transformation steps
- **Dmitrii Tochilkin**: symmetry integration
- **Alex Spirin**: warp and custom model support
- **KaliYuga**: Pixel Art Diffusion, Watercolor Diffusion and Pulp SciFi Diffusion models
- **Palmweaver / Chris Scalf**: OpenCLIP integration and the first KaliYuga model integration
- **Felipe3DArtist**: portrait_generator_v001
- **Maxwell Ingham**: copyright holder in the Disco Diffusion license

Nano's animation in `app/animate.py` is new code, but it follows their ideas and setting names:
keyframes, 2D and 3D camera moves, Video Input and Turbo. Nano doesn't use all of the work above.
Everyone is listed because the project they built is what made this one possible.

## Code it runs on

| Project | Authors | License |
|---|---|---|
| [guided-diffusion](https://github.com/openai/guided-diffusion), through the [kostarion fork](https://github.com/kostarion/guided-diffusion) | OpenAI, with changes by the fork's contributors | [MIT](LICENSES/guided-diffusion.txt) |
| [CLIP](https://github.com/openai/CLIP) | OpenAI | [MIT](LICENSES/clip.txt) |
| [ResizeRight](https://github.com/assafshocher/ResizeRight) | Assaf Shocher | [MIT](LICENSES/resize-right.txt) |
| [LPIPS](https://github.com/richzhang/PerceptualSimilarity) | Richard Zhang and others | [BSD 2-Clause](LICENSES/lpips.txt) |
| [Spandrel](https://github.com/chaiNNer-org/spandrel) | chaiNNer contributors | MIT |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Xintao Wang and others | BSD 3-Clause |
| [PyTorch](https://pytorch.org) | The PyTorch team | BSD-style |
| [TAESD](https://github.com/madebyollin/taesd) | Ollin Boer Bohan | MIT |
| [Tiny SD](https://huggingface.co/segmind/tiny-sd), an optional download. `app/prompt_graft.pt` was fitted from its features and the fine-tuned 512x512 model's | Segmind | CreativeML OpenRAIL-M |
| [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), an optional download | Lihe Yang and others | Apache 2.0 |
| [LaMa](https://github.com/advimman/lama) (big-lama), an optional download for Erase and Inpaint's Over a cleared area, as TorchScript from [Sanster/models](https://github.com/Sanster/models) | Roman Suvorov and others; Sanster | Apache 2.0 |
| [Robust CLIP (FARE)](https://github.com/chs20/RobustVLM) checkers, optional downloads for Robust checker | Christian Schlarmann, Naman Deep Singh, Francesco Croce and Matthias Hein | MIT |
| [OpenCLIP](https://github.com/mlfoundations/open_clip), which loads the robust checkers | OpenCLIP contributors | MIT |
| [Diffusers](https://github.com/huggingface/diffusers), [Transformers](https://github.com/huggingface/transformers) | Hugging Face | Apache 2.0 |
| [PyAV](https://github.com/PyAV-Org/PyAV), which reads and writes video. Its wheels include FFmpeg, built with x264 | PyAV contributors; the FFmpeg and x264 developers | BSD 3-Clause; FFmpeg and x264 under the GPL |
| [OpenCV](https://github.com/opencv/opencv), which follows a video's motion | OpenCV contributors | Apache 2.0 |
| [RIFE](https://github.com/hzwer/Practical-RIFE) 4.7, for in-between frames, an optional download. `app/rife.py` follows its network, as packaged by [vs-rife](https://github.com/HolyWu/vs-rife) | Zhewei Huang (hzwer) and others; HolyWu | [MIT](LICENSES/rife.txt) |
| [Keypoint R-CNN and Mask R-CNN](https://github.com/pytorch/vision), through torchvision, optional downloads for Pose control and Person control | The torchvision team | BSD 3-Clause |
| [RetinaFace](https://arxiv.org/abs/1905.00641) ResNet-50, an optional download for Face control and Face detail. `app/retinaface.py` follows the network of [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface), and the weights come from [facexlib](https://github.com/xinntao/facexlib)'s release | Jiankang Deng and others; biubug6; Xintao Wang | [MIT](LICENSES/pytorch-retinaface.txt); MIT |
| [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn) | Sebastián Ramírez; Encode | MIT; BSD 3-Clause |

guided-diffusion, CLIP, ResizeRight and the Python packages are downloaded by `setup.bat`. The
Real-ESRGAN x4plus model downloads from its authors' release if you choose it in the app. None of
these are part of this repository.

## Fonts

Plus Jakarta Sans, JetBrains Mono and Space Grotesk are under the SIL Open Font License. Their
licenses are in `app/static/fonts`.

## Thanks

[Rodrigo Barraza](https://github.com/rodrigo-barraza)'s
[Pristine Disco Diffusion](https://github.com/rodrigo-barraza/pristine-disco-diffusion) notebook
inspired this project and showed how good Disco Diffusion can look. Several default values in
`app/settings.py`, such as the guidance scales, started from its settings. It has no license on
GitHub, so none of its code or text is included here.

[neodisco](https://github.com/cobanov/neodisco) by cobanov showed where unrepeatable renders come
from, and that `range_scale` has no effect in the original code. That work led to Exact repeat. None
of its code is included here.

The tile mode averages overlapping views of one picture at every step, the idea behind
[MultiDiffusion](https://arxiv.org/abs/2302.08113) by Bar-Tal, Yariv, Lipman and Dekel. The app's
code for it was written for Nano. [Ultimate SD Upscale](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111)
showed how useful tiled work is. It is GPL-3 licensed, and none of its code is included here.

[Deforum Stable Diffusion](https://github.com/deforum-art/deforum-stable-diffusion), MIT licensed,
showed several animation ideas Nano uses: matching each frame's colors to the first in Lab,
sharpening each moved frame, blending the frames between Turbo's repaints, and moving the last
frame along the input video's optical flow. Nano's code for them was written for Nano. The Deforum
extension for the AUTOMATIC1111 web UI is AGPL-3, and none of its code was read or used.

The Disco Diffusion community, for years of prompts, settings and shared knowledge.

## Models

Nano downloads model weights the first time they are needed. They are not part of this repository,
and each has its own authors and terms. Check the terms of any model before using its output
commercially, especially the style models. A Handoff checkpoint is one you supply, and its own
license applies to it and to what it makes.
