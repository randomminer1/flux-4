import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
import gc,copy
from PIL.Image import Image
import threading,time
from pipelines.models import TextToImageRequest
import torch
from torchao.quantization import int8_weight_only,quantize_
from torch import Generator
from diffusers import FluxPipeline,FluxTransformer2DModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
from transformers import T5EncoderModel,T5TokenizerFast,CLIPTokenizer,CLIPTextModel
a="black-forest-labs/FLUX.1-schnell"
def b():gc.collect()
def c(d):
 if d.get("e")==None:d["e"]=CLIPTextModel.from_pretrained(a,subfolder="text_encoder",low_cpu_mem_usage=True,torch_dtype=torch.bfloat16)
 if d.get("f")==None:d["f"]=T5EncoderModel.from_pretrained(a,subfolder="text_encoder_2",low_cpu_mem_usage=True,torch_dtype=torch.bfloat16)
 if d.get("g")==None:d["g"]=CLIPTokenizer.from_pretrained(a,subfolder="tokenizer")
 if d.get("h")==None:d["h"]=T5TokenizerFast.from_pretrained(a,subfolder="tokenizer_2")
def i(j):
 if j.get("k")==None:
  j["k"]=FluxTransformer2DModel.from_pretrained("/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/transformer",low_cpu_mem_usage=True,torch_dtype=torch.bfloat16)
  quantize_(j["k"],int8_weight_only(),device="cuda")
def l(m):
 if m.get("n")==None:m["n"]=AutoencoderKL.from_pretrained(a,subfolder="vae",torch_dtype=torch.bfloat16).to("cuda")
 if m.get("o")==None and m.get("n") is not None:m["o"]=2**len(m["n"].config.block_out_channels)
 if m.get("p")==None and m.get("o") is not None:m["p"]=VaeImageProcessor(vae_scale_factor=m["o"])
def q(r):c(r);l(r);return r
def s(t,u):
 v=FluxPipeline.from_pretrained(a,text_encoder=u["e"],text_encoder_2=u["f"],tokenizer=u["g"],tokenizer_2=u["h"],transformer=None,vae=None);v.to('cuda')
 with torch.no_grad():w,x,y=v.encode_prompt(prompt=t,prompt_2=None,max_sequence_length=256)
 u["e"]=None;u["f"]=None;u["g"]=None;u["h"]=None
 return w,x,y
def z(aa,ab,ac,ad,ae,af):
 ag=FluxPipeline.from_pretrained(a,transformer=af["k"],text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,vae=None)
 ah=Generator(ag.device).manual_seed(ae) if ae is not None else None
 ai=ag(prompt_embeds=aa,pooled_prompt_embeds=ab,num_inference_steps=4,guidance_scale=0.0,width=ac,height=ad,generator=ah,output_type="latent").images
 af["k"]=None
 return ai
def aj(ak,al,am,an):
 ao=an["n"].to("cuda");ap=an["o"];aq=an["p"];am=am or 64*ap;al=al or 64*ap
 with torch.no_grad():
  ak=FluxPipeline._unpack_latents(ak,am,al,ap);ak=(ak/ao.config.scaling_factor)+ao.config.shift_factor
  ar=ao.decode(ak,return_dict=False)[0];ar=aq.postprocess(ar,output_type="pil")
 return ar
def n(at,au):
 av,aw=au;b()
 if "ax" in aw:aw.pop("ax").join()
 ay,az,ba=s(at.prompt,av)
 b()
 i(av)
 bb=z(ay,az,at.width,at.height,at.seed,av)
 bc=threading.Thread(target=c,args=(av,));aw["ax"]=bc;bc.start()
 b()
 bd=aj(bb,at.width,at.height,av)
 return bd[0]
def o():
 bf={};bg={};bf=q(bf)
 for _ in range(3):n(TextToImageRequest(prompt=""),(bf,bg))
 return (bf,bg)
