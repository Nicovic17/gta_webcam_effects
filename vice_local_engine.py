"""GPU-only StreamDiffusion backend, loaded exclusively by isolated Python 3.10."""
from pathlib import Path
import sys

import cv2
import numpy as np
from vice_local_settings import DEFAULT_PROMPT, normalize_prompt

STRENGTHS = {1: 42, 2: 35, 3: 28}


class StreamEngine:
    def __init__(self, root, *, width=512, height=320, download=False, strength=2,
                 prompt=DEFAULT_PROMPT):
        self.prompt = normalize_prompt(prompt)
        self.root = Path(root)
        source = self.root/'StreamDiffusion'/'src'
        if not source.is_dir():
            raise RuntimeError('StreamDiffusion mancante. Esegui SETUP_VICE_LOCALE.ps1.')
        sys.path.insert(0, str(source))
        import torch
        from diffusers import AutoencoderTiny, StableDiffusionPipeline
        from streamdiffusion import StreamDiffusion
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA non disponibile nel worker locale.')
        self.torch = torch
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        self.width, self.height = width, height
        self.acceleration = 'pytorch-sdpa'
        cache = str(self.root/'models')
        pipe = StableDiffusionPipeline.from_pretrained(
            'stabilityai/sd-turbo', variant='fp16', torch_dtype=torch.float16,
            use_safetensors=True, local_files_only=not download, cache_dir=cache,
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
            token=False).to('cuda')
        pipe.vae = AutoencoderTiny.from_pretrained(
            'madebyollin/taesd', torch_dtype=torch.float16, use_safetensors=True,
            local_files_only=not download, cache_dir=cache, token=False).to('cuda')
        try:
            pipe.enable_xformers_memory_efficient_attention()
            self.acceleration = 'xformers'
        except (ImportError, RuntimeError, ValueError):
            pass
        self.stream = StreamDiffusion(pipe, t_index_list=[STRENGTHS[strength]],
            torch_dtype=torch.float16, width=width, height=height,
            frame_buffer_size=1, cfg_type='none', use_denoising_batch=True)
        self.strength = None
        self.set_strength(strength)
        self.info = dict(model='stabilityai/sd-turbo', vae='madebyollin/taesd',
                         gpu=torch.cuda.get_device_name(0), width=width, height=height,
                         acceleration=self.acceleration, torch=torch.__version__,
                         cuda=torch.version.cuda)

    def set_strength(self, strength):
        self.configure(strength)

    def configure(self, strength, prompt=None):
        if strength not in STRENGTHS:
            raise ValueError('Intensita locale non valida.')
        prompt = self.prompt if prompt is None else normalize_prompt(prompt)
        if strength != self.strength:
            self.stream.t_list = [STRENGTHS[strength]]
            self.stream.prepare(prompt, num_inference_steps=50,
                                guidance_scale=1., seed=42)
            self.strength = strength
        elif prompt != self.prompt:
            # Upstream's live demo updates text embeddings only. The model,
            # scheduler and fixed noise stay resident between prompt edits.
            self.stream.update_prompt(prompt)
        self.prompt = prompt

    def infer(self, bgr, strength=2, *, prompt=None):
        self.configure(strength, prompt)
        h, w = bgr.shape[:2]
        ratio = min(self.width/w, self.height/h)
        rw, rh = max(1, round(w*ratio)), max(1, round(h*ratio))
        small = cv2.resize(bgr, (rw, rh), interpolation=cv2.INTER_AREA)
        left, top = (self.width-rw)//2, (self.height-rh)//2
        padded = cv2.copyMakeBorder(small, top, self.height-rh-top,
                                   left, self.width-rw-left, cv2.BORDER_REFLECT_101)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        torch = self.torch
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
            tensor = tensor.to(device='cuda', dtype=torch.float16)/127.5-1
            # The standard one-step StreamDiffusion path, without PIL copies or
            # extra per-frame CUDA event allocation/synchronization.
            latent = self.stream.encode_image(tensor)
            predicted = self.stream.predict_x0_batch(latent)
            decoded = self.stream.decode_image(predicted)[0]
            pixels = ((decoded.permute(1, 2, 0).clamp(-1, 1)+1)*127.5).round().to(torch.uint8).cpu().numpy()
        result = cv2.cvtColor(pixels[top:top+rh, left:left+rw], cv2.COLOR_RGB2BGR)
        # A little original signal steadies fine facial details; inference
        # remains generative, with a fixed noise seed across successive frames.
        return cv2.addWeighted(result, .90, small, .10, 0)

    def memory_mb(self):
        return round(self.torch.cuda.max_memory_allocated()/1024**2, 1)
