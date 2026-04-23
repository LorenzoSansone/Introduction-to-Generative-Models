# -*- coding: utf-8 -*-
"""ControlNet_Flow_quantized (9).ipynb

# ControlNet + FLUX.1-dev  (NF4 4-bit quantized)

**ControlNet** is a lightweight adapter that enables controllable image generation — for example, generating a scene that follows the lines of a sketch or matches a depth map. It works by attaching small "zero-convolution" layers to a frozen base model and training only those layers, so the original weights are never modified.

A ControlNet is conditioned on extra visual "structural controls" (canny edges, depth maps, human pose, etc.) that can be combined with a text prompt to steer the output toward the desired composition.

**This notebook adds NF4 4-bit quantization** (via `bitsandbytes`) so the full FLUX.1-dev pipeline fits in a free-tier Colab T4 (15 GB VRAM).

## 1 · Requirements

Before running large models like FLUX + ControlNet it is essential to verify that a GPU is active.

- **Why it matters**: GPUs can execute the parallel matrix operations in deep learning up to 100× faster than a CPU.
- **The check**: `torch.cuda.is_available()` confirms the runtime is connected to an accelerator.
- **Action required**: If the cell below prints `False`, go to *Runtime → Change runtime type* and select **T4 GPU** (or any available GPU).

We also install the quantization dependencies here.
"""

# Install required libraries.
# pip install -q diffusers transformers accelerate bitsandbytes sentencepiece protobuf
# pip install --upgrade diffusers huggingface_hub
# pip install matplotlib

import os
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers.utils import load_image
import matplotlib.pyplot as plt
from transformers import BitsAndBytesConfig
import gc
from diffusers import (
    FluxControlNetPipeline,
    FluxControlNetModel,
    FluxTransformer2DModel,
)
import time, gc
import os
from datetime import datetime

RES_DIR = "res"

os.environ["HF_TOKEN"] = ""  # replace with your Hugging Face token if you access private models or want to avoid rate limits

os.makedirs(RES_DIR, exist_ok=True)
print(f"Directory '{RES_DIR}' is ready.")


if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU  : {gpu_name}")
    print(f"VRAM : {vram_gb:.1f} GB")
    if vram_gb < 14:
        print("⚠  Less than 14 GB detected — NF4 quantization is essential here.")
    else:
        print("✅ Sufficient VRAM for NF4 quantization.")
else:
    print("GPU not found. Go to Runtime → Change runtime type and select a GPU.")

"""
## 2 · Prepare the control image (Canny edges)

We download a reference image and extract its **Canny edge map** with OpenCV. 
This edge map will be passed to the ControlNet as a *structural condition*, telling the model which outlines to follow during generation.

**Canny edge detection** works by:
1. Smoothing the image to reduce noise (Gaussian blur, implicit in OpenCV's implementation).
2. Computing gradient magnitudes and directions.
3. Keeping only pixels where the gradient is a local maximum (*non-maximum suppression*).
4. Applying two thresholds (`low_threshold`, `high_threshold`) to discard weak edges and keep strong ones.

The result is a binary (black/white) image. We replicate the single channel three times to match the 3-channel (RGB) input format expected by the model.
"""

# Download the reference image from the Hugging Face documentation dataset
original_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images"
    "/resolve/main/diffusers/non-enhanced-prompt.png"
)

# Convert PIL image to a NumPy array so OpenCV can process it
image = np.array(original_image)

# Canny thresholds: pixels with gradient above high_threshold are strong edges;
# pixels between the two thresholds are kept only if connected to a strong edge.
low_threshold  = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)

# cv2.Canny returns a single-channel (grayscale) image.
# Most deep learning pipelines — including ControlNet — expect 3-channel (RGB) input,
# so we stack the same channel three times.
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

print(f"Canny image shape: {np.shape(canny_image)}")

# Display the original image alongside the extracted Canny edge map
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Reference image and its Canny edge map", fontsize=14, color = "blue")

axes[0].set_title("Original image")
axes[0].imshow(original_image)
axes[0].axis("off")

axes[1].set_title("Canny edge map (control signal)")
axes[1].imshow(canny_image)
axes[1].axis("off")

plt.tight_layout()
plt.savefig(f'./{RES_DIR}/canny_comparison.png')
plt.show()

"""
## 3 · Quantization configuration

FLUX.1-dev is a **12 B parameter DiT** (Diffusion Transformer). Loaded in `bfloat16` it requires roughly 24 GB of VRAM — far beyond the T4's 15 GB (if you run in Colab). 
We use two techniques to fit:

| Component | Strategy | VRAM saved |
|-----------|----------|------------|
| DiT transformer | NF4 4-bit (bitsandbytes) | ~14 GB → ~7 GB |
| T5-XXL text encoder | CPU offload | freed from GPU entirely |
| CLIP-L text encoder | stays on GPU (small, ~250 MB) | — |
| ControlNet adapter | bfloat16, small (~1.5 GB) | — |

**NF4 (NormalFloat4)** distributes its 16 quantization levels optimally for weights that follow a normal (Gaussian) distribution — which transformer weights typically do. Enabling `double_quant` additionally quantizes the scale constants themselves, saving a further ~0.37 bits per parameter (~400 MB on the full transformer).

**CPU offload** (`enable_model_cpu_offload`) moves each sub-model to the GPU only while it is actively computing, then immediately moves it back to CPU RAM. This keeps peak GPU usage low at the cost of slightly slower inference.
"""

# NF4 4-bit quantization configuration for the FLUX DiT transformer.
# - load_in_4bit: compress each weight to 4 bits (from 16), halving memory use.
# - bnb_4bit_quant_type='nf4': use the NormalFloat4 data type, which minimises
#   quantization error for Gaussian-distributed weights.
# - bnb_4bit_compute_dtype: use bfloat16 for the actual arithmetic during the
#   forward pass (weights are dequantized on-the-fly to this dtype).
# - bnb_4bit_use_double_quant: apply a second quantization to the scale factors,
#   saving roughly 0.37 additional bits per parameter.

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("NF4 quantization config ready.")

"""
## 4 · Load models

We load three components:

1. **`FluxTransformer2DModel`** — the DiT backbone of FLUX.1-dev. This is the heaviest piece (~12 GB in bf16) and is loaded with NF4 quantization (~7 GB).
2. **`FluxControlNetModel`** (`InstantX/FLUX.1-dev-Controlnet-Canny`) — a ControlNet adapter trained specifically on FLUX.1-dev with Canny edge conditioning. It is small (~1.5 GB) and loaded in `bfloat16`.
3. **`FluxControlNetPipeline`** — the full pipeline that wires together the quantized transformer, the ControlNet, the two text encoders (CLIP-L and T5-XXL), the VAE, and the scheduler.
"""

# Free any leftover GPU memory from previous cells before loading large models
gc.collect()
torch.cuda.empty_cache()

# ── Step 1: load the ControlNet adapter (bfloat16, small enough to fit as-is) ──
# InstantX/FLUX.1-dev-Controlnet-Canny is an adapter trained on FLUX.1-dev
# that conditions generation on Canny edge maps.
print("Loading ControlNet adapter...")
controlnet = FluxControlNetModel.from_pretrained(
    "InstantX/FLUX.1-dev-Controlnet-Canny",
    torch_dtype=torch.bfloat16,
)

# ── Step 2: load the FLUX DiT transformer with NF4 4-bit quantization ──
# By quantizing this component separately we avoid loading the full 24 GB pipeline
# in fp16/bf16 and then trying to quantize it in-place (which would require 2×memory).
# Loading it directly in 4-bit means peak usage never exceeds ~8–10 GB.
print("Loading FLUX transformer (NF4 4-bit)...")
transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=nf4_config,  # apply NF4 to every Linear layer
    torch_dtype=torch.bfloat16,
)

# ── Step 3: assemble the full pipeline, injecting the quantized transformer ──
# The remaining components (CLIP-L, T5-XXL, VAE) are loaded in bfloat16.
# We do NOT call .to('cuda') here — instead we use enable_model_cpu_offload()
# so each sub-model is moved to the GPU only when needed, keeping peak VRAM low.
print("Assembling pipeline...")
pipeline = FluxControlNetPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
)

# CPU offload: each model block is transferred to the GPU only for its forward pass,
# then immediately moved back to CPU. This is slower than a full .to('cuda') but
# essential when total model size exceeds available VRAM.
pipeline.enable_model_cpu_offload()

# Attention slicing splits the attention computation into smaller chunks,
# trading a little speed for lower peak memory during the attention layers.
pipeline.enable_attention_slicing()

used_gb = torch.cuda.memory_allocated() / 1024**3
print(f"\n✅ Pipeline ready. GPU memory allocated: {used_gb:.2f} GB")

"""## 5 · Generate an image

Key pipeline parameters:

| Parameter | Role |
|-----------|------|
| `prompt` | Text description guiding the image content |
| `control_image` | The Canny edge map — the structural condition |
| `controlnet_conditioning_scale` | How strongly the ControlNet overrides the base model (0 = ignore, 1 = full control) |
| `num_inference_steps` | Number of denoising steps; more steps → higher quality, slower inference |
| `guidance_scale` | How closely the output follows the text prompt (higher = more literal, less creative) |
| `height` / `width` | Output resolution; reduce to 512×512 if you run out of memory |
"""

# ── Generation parameters ──────────────────────────────────
PROMPT = (
    "A photorealistic overhead image of a cat reclining sideways "
    "in a flamingo pool floatie holding a margarita. "
    "The cat is floating leisurely in the pool, completely relaxed and happy."
)
HEIGHT                      = 768   # reduce to 512 if OOM
WIDTH                       = 768
NUM_STEPS                   = 50   # 20–50 works well for FLUX
GUIDANCE_SCALE              = 3.5   # typical range: 2.0–7.0
CONTROLNET_SCALE            = 0.5   # 0.3–0.7 is a good starting range
SEED                        = 42
# ──────────────────────────────────────────────────────────────────────────────

gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

generator = torch.Generator(device="cpu").manual_seed(SEED)

t0 = time.time()
print(f"Generating {WIDTH}×{HEIGHT} image with {NUM_STEPS} steps...")

with torch.inference_mode():
    result = pipeline(
        prompt=PROMPT,
        control_image=canny_image,
        controlnet_conditioning_scale=CONTROLNET_SCALE,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
        max_sequence_length=512,  # FLUX uses T5 which supports up to 512 tokens
    )

elapsed   = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / 1024**3
print(f"Done in {elapsed:.1f}s  |  peak VRAM: {peak_vram:.2f} GB")

output_image = result.images[0]
plt.figure(figsize=(8, 8))
plt.suptitle(f"Generated image controlnet_conditioning_scale={CONTROLNET_SCALE}, guidance_scale={GUIDANCE_SCALE}", fontsize=14, color = "blue")
plt.imshow(output_image)
plt.axis("off")
plt.tight_layout()
plt.savefig(f'./{RES_DIR}/generated_image.png')
plt.show()

"""

## 6 · Compare `controlnet_conditioning_scale` values

The `controlnet_conditioning_scale` parameter controls how much the Canny edge map influences the final image. A value close to **0** almost ignores the edges (freeform generation); a value close to **1** forces the output to strictly follow every edge in the map.

The loop below generates one image per scale value so you can compare the trade-off visually.

"""


# Scale values to compare — from loose to strict edge adherence
scale_values = [0, 0.25, 0.5, 0.75, 1.0]
sweep_results = []

for scale in scale_values:
    print(f"Generating with controlnet_conditioning_scale={scale}...")
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        out = pipeline(
            prompt=PROMPT,
            control_image=canny_image,
            controlnet_conditioning_scale=scale,  # the variable we are sweeping
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            max_sequence_length=512,
        )
    sweep_results.append((scale, out.images[0]))

# Display all results as a single grid
fig, axes = plt.subplots(1, len(sweep_results), figsize=(5 * len(sweep_results), 5))
fig.suptitle("Effect of controlnet_conditioning_scale", fontsize=14, color = "blue")
for ax, (scale, img) in zip(axes, sweep_results):
    ax.imshow(img)
    ax.set_title(f"scale={scale}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f'./{RES_DIR}/comparison_conditioning_scale.png')
plt.show()

"""
## 7 · Compare `guidance_scale` values

The `guidance_scale` (often called "CFG scale") controls how strongly the image generation is steered by the text prompt. 
Lower values allow more creative freedom, while higher values produce images that more closely match the prompt but may be less diverse. 
The loop below generates one image per guidance scale value so you can compare the trade-off visually.
"""
# Guidance scale values to compare — from low to high CFG
guidance_values = [1.0, 3.5, 5.0, 7.5, 10.0]
guidance_results = []

for gs in guidance_values:
    print(f"Generating with guidance_scale={gs}...")
    gc.collect()
    torch.cuda.empty_cache()

    with torch.inference_mode():
        out = pipeline(
            prompt=PROMPT,
            control_image=canny_image,
            controlnet_conditioning_scale=CONTROLNET_SCALE,  # fixed
            num_inference_steps=NUM_STEPS,
            guidance_scale=gs,                  # the variable we are sweeping
            height=HEIGHT,
            width=WIDTH,
            generator=torch.Generator(device="cpu").manual_seed(SEED),
            max_sequence_length=512,
        )
    guidance_results.append((gs, out.images[0]))

# Display all results as a single grid
fig, axes = plt.subplots(1, len(guidance_results), figsize=(5 * len(guidance_results), 5))
fig.suptitle("Effect of guidance_scale", fontsize=14, color = "blue")
for ax, (gs, img) in zip(axes, guidance_results):
    ax.imshow(img)
    ax.set_title(f"guidance_scale={gs}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(f'./{RES_DIR}/comparison_guidance_scale.png')
plt.show()

"""
## Bibliography

- **ControlNet (Diffusers)**: https://huggingface.co/docs/diffusers/using-diffusers/controlnet
- **FluxControlNetModel**: https://huggingface.co/docs/diffusers/api/models/controlnet_flux
- **FluxControlNetPipeline**: https://huggingface.co/docs/diffusers/api/pipelines/controlnet_flux
- **InstantX FLUX Canny ControlNet**: https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Canny
- **BitsAndBytes NF4**: https://huggingface.co/blog/4bit-transformers-bitsandbytes
- **FLUX.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **BitsAndBytes Quantization**: https://huggingface.co/docs/diffusers/quantization/bitsandbytes

"""