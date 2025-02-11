import gc, os, torch
from PIL.Image import Image as _I
from diffusers import FluxPipeline as _P, FluxTransformer2DModel as _T, AutoencoderKL as _V
from huggingface_hub.constants import HF_HUB_CACHE as _H
from pipelines.models import TextToImageRequest as _R
from torch import Generator as _G
from transformers import T5EncoderModel as _E, CLIPTextModel as _C
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe as _a

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
_c, _r = "black-forest-labs/FLUX.1-schnell", "741f7c3ce8b383c54771c7003378a50191e9efe9"

def _f() -> _P:
    _x = _C.from_pretrained(_c, revision=_r, subfolder="text_encoder", local_files_only=True, torch_dtype=torch.bfloat16)
    _y = _E.from_pretrained(_c, revision=_r, subfolder="text_encoder_2", local_files_only=True, torch_dtype=torch.bfloat16)
    _z = _V.from_pretrained(_c, revision=_r, subfolder="vae", local_files_only=True, torch_dtype=torch.bfloat16)
    _p = os.path.join(_H, "models--barneystinson--FLUX.1-schnell-int8wo/snapshots/b9fa75333f9319a48b411a2618f6f353966be599")
    _t = _T.from_pretrained(_p, torch_dtype=torch.bfloat16, use_safetensors=False)
    _pipe = _P.from_pretrained(
        _c, revision=_r, local_files_only=True,
        text_encoder=_x, text_encoder_2=_y,
        transformer=_t, vae=_z, torch_dtype=torch.bfloat16
    ).to("cuda")
    _a(_pipe, residual_diff_threshold=0.5)
    _pipe.vae = torch.compile(_pipe.vae, mode="max-autotune")
    _pipe("")
    return _pipe

def _g(_req: _R, _pipe: _P) -> _I:
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    _gen = _G(_pipe.device).manual_seed(_req.seed)
    return _pipe(
        _req.prompt, generator=_gen, guidance_scale=0.0,
        num_inference_steps=4, max_sequence_length=256,
        height=_req.height, width=_req.width
    ).images[0]
