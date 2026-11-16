"""What makes the renderer fit and run quickly on an 8GB card. Used by renderer.py.

Disco Diffusion's memory problem is the opposite of Stable Diffusion's. Weights are only about 1GB.
The rest is the full-size working data kept for the guidance backward pass, so that is what these
target.
"""
import importlib
import os

import torch
import torch.nn.functional as F

GB = 2**30
_PATCHED = set()  # the patches already made in this process
VRAM_HEADROOM_GB = 0.75  # left free for Windows to draw the screens; see limit_vram
OFFLOAD_MIN_ELEMENTS = 5_000_000


def half_precision_norm():
    """GroupNorm32 casts to 32-bit to normalize, doubling the memory of a whole layer.

    At full size that one cast is the allocation that runs an 8GB card out of memory, so large
    layers stay 16-bit. Smaller layers keep the original behavior: they cost little, and it keeps
    this change away from the deep layers that shape composition.
    """
    from guided_diffusion import nn as gd_nn
    if "norm" in _PATCHED:  # a worker that stays open renders many times, and a second patch would wrap the first
        return
    _PATCHED.add("norm")

    def forward(self, x):
        if x.dtype == torch.float16 and x.shape[-1] >= 128:
            return F.group_norm(x, self.num_groups, self.weight.half(), self.bias.half(), self.eps)
        return F.group_norm(x.float(), self.num_groups, self.weight, self.bias, self.eps).type(x.dtype)

    gd_nn.GroupNorm32.forward = forward


def _half_norm(x):
    """True when half_precision_norm normalizes this tensor in 16-bit with F.group_norm."""
    return x.dtype == torch.float16 and x.dim() == 4 and x.shape[-1] >= 128 and x.is_contiguous()


def _norm_stats(x, norm):
    """GroupNorm through the same kernel F.group_norm uses, also returning what its backward needs."""
    n, c = x.shape[:2]
    weight = norm.weight.half()
    out, mean, rstd = torch.ops.aten.native_group_norm(x, weight, norm.bias.half(), n, c, x[0, 0].numel(),
                                                       norm.num_groups, norm.eps)
    return out, (mean, rstd, weight)


def _norm_backward(grad, x, stats, norm):
    """The input gradient of GroupNorm, as autograd computes it when the norm's weights are frozen."""
    mean, rstd, weight = stats
    n, c = x.shape[:2]
    return torch.ops.aten.native_group_norm_backward(grad, x, mean, rstd, weight, n, c, x[0, 0].numel(),
                                                     norm.num_groups, [True, False, False])[0]


class _Join(torch.autograd.Function):
    """torch.cat on channels whose backward hands back two separate gradients, not views of one.

    The skip connection's half of the gradient waits until the backward pass reaches the encoder.
    As a view it would keep the whole joined gradient alive, 0.5GB per block at 1280x768.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.split = a.shape[1]
        return torch.cat([a, b], dim=1)

    @staticmethod
    def backward(ctx, grad):
        return grad[:, :ctx.split].clone(), grad[:, ctx.split:].clone()  # .contiguous() would return views


def _silu_grad(grad, pre):
    """SiLU's input gradient, written over grad when it can be (the kernel autograd uses)."""
    if grad.is_contiguous():
        return torch.ops.aten.silu_backward.grad_input(grad, pre, grad_input=grad)
    return torch.ops.aten.silu_backward(grad, pre)


def _norm_silu_grad(grad, x, norm, scale=None, shift=None):
    """Input gradient of silu(norm(x)), or of silu(norm(x) * (1 + scale) + shift). May overwrite grad.

    Recomputes the norm, drops its output once SiLU's gradient is taken, then runs GroupNorm's
    backward: three tensors of x's size at most, against five when autograd keeps everything.
    """
    if _half_norm(x):
        out, stats = _norm_stats(x, norm)
        if scale is not None:
            out.mul_(1 + scale).add_(shift)
        grad = _silu_grad(grad, out)
        del out
        if scale is not None:
            grad = grad.mul_(1 + scale)
        return _norm_backward(grad, x, stats, norm)
    with torch.enable_grad():  # the small 32-bit layers: plain autograd
        xr = x.detach().requires_grad_()
        y = norm(xr)
        if scale is not None:
            y = y * (1 + scale) + shift
        return torch.autograd.grad(F.silu(y), xr, grad)[0]


def _conv_grad(conv, grad, shape):
    """A frozen convolution's input gradient, which needs the weights but not the input."""
    return torch.nn.grad.conv2d_input(shape, conv.weight, grad, conv.stride, conv.padding, conv.dilation, conv.groups)


def _scale_shift(block, emb, like):
    emb_out = block.emb_layers(emb).type(like.dtype)
    while len(emb_out.shape) < len(like.shape):
        emb_out = emb_out[..., None]
    return torch.chunk(emb_out, 2, dim=1)


def _block_forward(block, x, emb):
    """guided-diffusion's ResBlock._forward without gradients, reusing buffers in place.

    An upsampling block adds its input straight onto the output instead of making a full-size copy.
    """
    from guided_diffusion import unet as gd_unet
    h = F.silu(block.in_layers[0](x), inplace=True)
    h = block.in_layers[2](block.h_upd(h))
    scale, shift = _scale_shift(block, emb, h)
    h = F.silu(block.out_layers[0](h).mul_(1 + scale).add_(shift), inplace=True)
    h = block.out_layers[3](h)
    if isinstance(block.x_upd, gd_unet.Upsample) and isinstance(block.skip_connection, torch.nn.Identity):
        n, c, height, width = x.shape
        h.view(n, c, height, 2, width, 2).add_(x.view(n, c, height, 1, width, 1))
        return h
    return h.add_(block.skip_connection(block.x_upd(x)))


class _LeanBlock(torch.autograd.Function):
    """A ResBlock that keeps only its input and steps back through itself by hand in the backward pass.

    Autograd would keep every intermediate, and it runs the skip connection's backward first and holds
    that gradient while the main path works. Here each buffer is freed as soon as it is used, the
    convolutions keep nothing (their weights are frozen), and the skip connection comes last.
    The kernels are the ones autograd calls, so the gradients are the same.
    """

    @staticmethod
    def forward(ctx, x, emb, block):
        ctx.block = block
        ctx.save_for_backward(x, emb)  # offloaded by the hooks around apply() when Offload is on
        return _block_forward(block, x, emb)

    @staticmethod
    def backward(ctx, grad_out):
        x, emb = ctx.saved_tensors
        b = ctx.block
        in_norm, in_conv, out_norm, out_conv = b.in_layers[0], b.in_layers[2], b.out_layers[0], b.out_layers[3]
        resized = not isinstance(b.h_upd, torch.nn.Identity)
        with torch.no_grad():
            a = F.silu(in_norm(x), inplace=True)
            u = b.h_upd(a)
            u_shape = u.shape
            h = in_conv(u)
            del u
            if not resized:
                del a
            scale, shift = _scale_shift(b, emb, h)
            g = _conv_grad(out_conv, grad_out, h.shape)
            g = _norm_silu_grad(g, h, out_norm, scale, shift)
            del h
            g = _conv_grad(in_conv, g, u_shape)
            if resized:
                with torch.enable_grad():
                    ar = a.detach().requires_grad_()
                    g = torch.autograd.grad(b.h_upd(ar), ar, g)[0]
                del a, ar
            dx = _norm_silu_grad(g, x, in_norm)
            del g
            if isinstance(b.x_upd, torch.nn.Identity) and isinstance(b.skip_connection, torch.nn.Identity):
                return dx.add_(grad_out), None, None
            if isinstance(b.x_upd, torch.nn.Identity):
                return dx.add_(_conv_grad(b.skip_connection, grad_out, x.shape)), None, None
            with torch.enable_grad():
                xr = x.detach().requires_grad_()
                skip = torch.autograd.grad(b.skip_connection(b.x_upd(xr)), xr, grad_out)[0]
            return dx.add_(skip), None, None


_OFFLOAD = None  # (pack, unpack) once offload_activations() has run


def lean_unet():
    """Runs the UNet's ResBlocks in much less memory, with the same arithmetic.

    At 1280x768 a full-size block holds 0.5-0.7GB per copy of its data, and the stock code keeps four or
    five copies alive at once in the forward pass and more in the guidance backward pass. See
    _block_forward, _LeanBlock and _Join. The UNet's weights must not require gradients (load_unet).
    """
    from guided_diffusion import unet as gd_unet
    if "lean" in _PATCHED:
        return
    _PATCHED.add("lean")

    stock_forward = gd_unet.ResBlock.forward

    def block_forward(self, x, emb):
        if not self.use_scale_shift_norm or x.dim() != 4 or self.use_conv:
            return stock_forward(self, x, emb)
        if torch.is_grad_enabled() and x.requires_grad:
            if _OFFLOAD:  # saved tensors are packed after forward returns, so the hooks wrap the call
                from torch.autograd.graph import saved_tensors_hooks
                with saved_tensors_hooks(*_OFFLOAD):
                    return _LeanBlock.apply(x, emb, self)
            return _LeanBlock.apply(x, emb, self)
        return _block_forward(self, x, emb)

    def unet_forward(self, x, timesteps, y=None):
        """guided-diffusion's UNetModel.forward, joining skip connections with _Join."""
        assert (y is not None) == (self.num_classes is not None)
        hs = []
        emb = self.time_embed(gd_unet.timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            skip = hs.pop()
            h = _Join.apply(h, skip) if torch.is_grad_enabled() else torch.cat([h, skip], dim=1)
            del skip
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)

    gd_unet.ResBlock.forward = block_forward
    gd_unet.UNetModel.forward = unet_forward


def offload_activations(on=True):
    """Keeps the UNet's saved activations in system RAM instead of VRAM, or with on=False stops doing so.

    They go through save_for_backward, so PyTorch's saved-tensor hooks can move the large ones to
    RAM and bring them back on demand, the same mechanism as torch.autograd.graph.save_on_cpu.
    A worker that stays open switches it for each render.
    """
    global _OFFLOAD
    if not on:
        _OFFLOAD = None  # the patched checkpoint and _LeanBlock both check this, so nothing is moved
        return
    from torch.autograd.graph import saved_tensors_hooks
    from guided_diffusion import nn as gd_nn

    original = gd_nn.CheckpointFunction

    def pack(t):
        if t.dim() == 4 and t.numel() >= OFFLOAD_MIN_ELEMENTS and not isinstance(t, torch.nn.Parameter):
            return t.device, t.to("cpu")
        return None, t

    def unpack(saved):
        device, t = saved
        return t.to(device) if device is not None else t

    _OFFLOAD = (pack, unpack)  # _LeanBlock saves its input the same way
    if "offload" in _PATCHED:
        return
    _PATCHED.add("offload")

    class OffloadingCheckpoint:
        @staticmethod
        def apply(func, length, *args):
            if _OFFLOAD is None or not any(a.requires_grad for a in args[:length]):
                return original.apply(func, length, *args)  # no backward pass, nothing to keep
            with saved_tensors_hooks(*_OFFLOAD):
                return original.apply(func, length, *args)

    gd_nn.CheckpointFunction = OffloadingCheckpoint


def clip_chunk(model):
    """How many cuts go through this CLIP model at once.

    ViT-L/14 holds about four times the working memory of the B models for every cut, and all 16
    cuts at once run an 8GB card out of memory at 512x512. Four at a time gives the same gradient,
    summed. The smaller models still take every cut at once.
    """
    transformer = getattr(model.visual, "transformer", None)
    return 4 if transformer is not None and transformer.width >= 1024 else 10**6


def park(clip_models):
    """CLIP is idle while the UNet runs backward, the point of highest VRAM use."""
    for model in clip_models:
        model.visual.to("cpu")


def unpark(clip_models, device):
    for model in clip_models:
        model.visual.to(device)


def free_text_encoders(clip_models):
    """Prompts are already encoded by now, so the text halves of CLIP can leave the GPU."""
    for model in clip_models:
        for part in ("transformer", "token_embedding", "ln_final"):
            module = getattr(model, part, None)
            if module is not None:
                module.cpu()
    torch.cuda.empty_cache()


def text_encoders_back(clip_models, device):
    """Brings the text halves of CLIP back for a moment, to encode prompts set during a render."""
    for model in clip_models:
        for part in ("transformer", "token_embedding", "ln_final"):
            module = getattr(model, part, None)
            if module is not None:
                module.to(device)


def limit_vram():
    """Caps this process below what is free, so running out is a clean error.

    Past a point, Windows starts paging video memory instead of refusing it: the GPU drops to idle
    clocks, renders stall for minutes and other GPU apps suffer. Failing early is much kinder.
    The headroom is for the desktop. With only 0.15GB, a 768x512 render can fill an 8GB card to
    7.9GB and the screens go gray. ComfyUI keeps 0.6GB free on Windows for the same reason; this
    keeps a little more because CUDA's own kernels load after the cap is set and aren't counted in it.
    Returns the cap in GB.
    """
    global cap_bytes
    free, total = torch.cuda.mem_get_info()
    # What this worker already holds is its own to use, not another app's. Counted as taken, it was taken twice:
    # once here and again in room(), which subtracts it from the cap.
    cap = free + torch.cuda.memory_reserved() - int(VRAM_HEADROOM_GB * GB)
    cap_bytes = max(cap, GB)
    torch.cuda.set_per_process_memory_fraction(cap_bytes / total)
    return cap / GB


cap_bytes = None


def room():
    """Bytes this process can still take on the GPU, under its cap and with the desktop's headroom kept."""
    free = torch.cuda.mem_get_info()[0] - int(VRAM_HEADROOM_GB * GB)
    if cap_bytes is None:
        return free
    return min(free, cap_bytes - torch.cuda.memory_reserved())


def load_vit_l14(clip_package, weight_paths):
    """Loads ViT-L/14 from a local safetensors copy when there is one, saving a 0.87GB download."""
    cached = os.path.expanduser("~/.cache/clip/ViT-L-14.pt")
    source = next((p for p in weight_paths if os.path.exists(p)), None)
    if os.path.exists(cached) or source is None:
        return clip_package.load("ViT-L/14", jit=False)[0]
    from safetensors.torch import load_file
    return importlib.import_module("clip.model").build_model(load_file(source))
