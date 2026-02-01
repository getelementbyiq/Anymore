# LTX-2 RunPod Deployment Guide

Deploy LTX-2 video generation on RunPod's serverless infrastructure.

## Overview

LTX-2 is a 19B parameter audio-video generation model. This deployment supports:
- **All 5 pipelines**: Distilled, TwoStage, OneStage, IC-LoRA, Keyframe
- **All LoRAs**: IC-LoRAs (Canny, Depth, Pose, Detailer) + Camera Controls
- **FP8 precision** for reduced memory usage
- **RunPod Serverless** for scalable, pay-per-use inference

---

## Hardware Requirements

| GPU | VRAM | Configuration | Performance |
|-----|------|---------------|-------------|
| H100/A100 | 80GB | Default | Best |
| A10G | 24GB | FP8 required | Good |
| RTX 4090 | 24GB | FP8 required | Good |
| A40 | 48GB | FP8 optional | Very Good |

**Minimum**: 24GB VRAM with FP8 enabled

---

## Available Models (~50-100GB)

### Main Checkpoints (choose one)
| Model | Size | Description |
|-------|------|-------------|
| `ltx-2-19b-distilled-fp8` | ~19GB | **Fastest, lowest memory** (recommended) |
| `ltx-2-19b-distilled` | ~38GB | Fast inference |
| `ltx-2-19b-dev-fp8` | ~19GB | Best quality, low memory |
| `ltx-2-19b-dev` | ~38GB | Best quality |

### Required Components
| Model | Size | Description |
|-------|------|-------------|
| `ltx-2-spatial-upscaler-x2-1.0` | ~500MB | 2x upscaling for two-stage pipelines |
| `ltx-2-19b-distilled-lora-384` | ~150MB | Required for TwoStage pipeline |
| `gemma-3-12b-it-qat-q4_0` | ~12GB | Text encoder |

### Optional LoRAs
| Category | LoRAs |
|----------|-------|
| **IC-LoRAs** | canny, depth, detailer, pose |
| **Camera** | dolly-in, dolly-out, dolly-left, dolly-right, jib-up, jib-down, static |

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/getelementbyiq/Anymore.git
cd Anymore
```

### 2. Download Models

```bash
pip install huggingface_hub

# Essential models only (~32GB)
python runpod/download_models.py --model-dir ./models

# All models including LoRAs (~50GB+)
python runpod/download_models.py --model-dir ./models --all

# Custom selection
python runpod/download_models.py --model-dir ./models \
    --variant distilled-fp8 \
    --ic-loras canny depth \
    --camera-loras

# List all available models
python runpod/download_models.py --list
```

### 3. Build & Push Docker Image

```bash
# Build
docker build -t ltx2-runpod -f runpod/Dockerfile .

# Push to Docker Hub
docker tag ltx2-runpod yourusername/ltx2-runpod:latest
docker push yourusername/ltx2-runpod:latest
```

### 4. Create RunPod Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Create **Network Volume** and upload models (recommended)
3. Create **Serverless Endpoint**:
   - **Image**: `yourusername/ltx2-runpod:latest`
   - **GPU**: A100 80GB or A10G 24GB
   - **Volume**: Mount at `/runpod-volume`
   - **Environment**:
     ```
     MODEL_DIR=/runpod-volume/models
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
     ```

---

## API Reference

### Check Status

```json
{
  "input": {
    "action": "status"
  }
}
```

Response:
```json
{
  "gpu": "NVIDIA A100-SXM4-80GB",
  "gpu_memory_gb": 80.0,
  "checkpoint": "ltx-2-19b-distilled-fp8.safetensors",
  "gemma": true,
  "available_loras": ["canny", "depth", "dolly-in", "static"],
  "available_pipelines": ["distilled", "two_stage", "one_stage", "ic_lora", "keyframe"]
}
```

### Text-to-Video (T2V)

```json
{
  "input": {
    "prompt": "A golden retriever running through a sunny meadow, slow motion, cinematic",
    "negative_prompt": "blurry, low quality",
    "seed": 42,
    "height": 544,
    "width": 960,
    "num_frames": 49,
    "frame_rate": 25.0,
    "pipeline": "distilled",
    "loras": ["static"],
    "enhance_prompt": false
  }
}
```

### Image-to-Video (I2V)

```json
{
  "input": {
    "prompt": "The scene comes to life with gentle movement",
    "pipeline": "two_stage",
    "images": [
      {
        "base64": "<base64-encoded-image>",
        "frame_index": 0,
        "strength": 1.0
      }
    ],
    "seed": 42,
    "height": 544,
    "width": 960,
    "num_frames": 49
  }
}
```

### Video-to-Video with IC-LoRA

```json
{
  "input": {
    "prompt": "Transform to anime style",
    "pipeline": "ic_lora",
    "loras": ["canny"],
    "reference_video": "<base64-encoded-video>",
    "seed": 42,
    "height": 544,
    "width": 960,
    "num_frames": 49
  }
}
```

### Keyframe Interpolation

```json
{
  "input": {
    "prompt": "Smooth transition between scenes",
    "pipeline": "keyframe",
    "images": [
      {"base64": "<frame1>", "frame_index": 0, "strength": 1.0},
      {"base64": "<frame2>", "frame_index": 48, "strength": 1.0}
    ],
    "num_frames": 49
  }
}
```

### With Camera Control

```json
{
  "input": {
    "prompt": "A beautiful landscape",
    "pipeline": "distilled",
    "loras": ["dolly-in"],
    "seed": 42
  }
}
```

### Response Format

```json
{
  "video_base64": "<base64-encoded-mp4>",
  "seed": 42,
  "prompt": "A golden retriever...",
  "pipeline": "distilled",
  "loras": ["static"],
  "status": "success"
}
```

---

## Parameters Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Text description |
| `negative_prompt` | string | "" | What to avoid |
| `seed` | int | 42 | Random seed |
| `height` | int | 544 | Height (multiple of 32) |
| `width` | int | 960 | Width (multiple of 32) |
| `num_frames` | int | 49 | Frames (1 + n*8) |
| `frame_rate` | float | 25.0 | FPS |

### Pipeline Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pipeline` | string | "distilled" | Pipeline type |
| `loras` | list | [] | LoRAs to apply |
| `fp8` | bool | true | Use FP8 precision |
| `num_inference_steps` | int | 8/40 | Denoising steps |
| `enhance_prompt` | bool | false | Auto-enhance prompt |

### Guidance Parameters (advanced)

```json
{
  "guidance": {
    "video_cfg_scale": 3.0,
    "video_stg_scale": 1.0,
    "audio_cfg_scale": 7.0,
    "audio_stg_scale": 1.0
  }
}
```

### Pipelines

| Pipeline | Steps | Best For | LoRA Support |
|----------|-------|----------|--------------|
| `distilled` | 8 | **Fastest inference** | Yes |
| `two_stage` | 40 | **Best quality** | Yes |
| `one_stage` | 40 | Prototyping | Yes |
| `ic_lora` | 40 | Video-to-video | Required |
| `keyframe` | 40 | Interpolation | Yes |

### Available LoRAs

| Name | Type | Effect |
|------|------|--------|
| `canny` | IC-LoRA | Edge-guided generation |
| `depth` | IC-LoRA | Depth-guided generation |
| `detailer` | IC-LoRA | Enhance/upscale |
| `pose` | IC-LoRA | Pose-guided generation |
| `dolly-in` | Camera | Zoom in |
| `dolly-out` | Camera | Zoom out |
| `dolly-left` | Camera | Pan left |
| `dolly-right` | Camera | Pan right |
| `jib-up` | Camera | Crane up |
| `jib-down` | Camera | Crane down |
| `static` | Camera | Locked camera |

---

## Python Client Example

```python
import runpod
import base64
from pathlib import Path

runpod.api_key = "your_api_key"
endpoint_id = "your_endpoint_id"

# Check status
status = runpod.run_sync(endpoint_id, {"action": "status"})
print(f"GPU: {status['gpu']}")
print(f"Available LoRAs: {status['available_loras']}")

# Generate video
result = runpod.run_sync(endpoint_id, {
    "prompt": "A beautiful sunset over the ocean, cinematic, 4K",
    "pipeline": "distilled",
    "loras": ["static"],
    "seed": 42,
    "height": 544,
    "width": 960,
    "num_frames": 49,
})

# Save video
if result["status"] == "success":
    video = base64.b64decode(result["video_base64"])
    Path("output.mp4").write_bytes(video)
    print(f"Saved! Size: {len(video) / 1024 / 1024:.1f} MB")
```

---

## Troubleshooting

### Out of Memory (OOM)
- Use `fp8: true`
- Reduce `height`/`width` or `num_frames`
- Use `distilled` pipeline
- Use A100 80GB

### Slow Cold Starts
- Pre-download models to Network Volume
- Set min workers > 0

### Model Not Found
- Check `MODEL_DIR` environment variable
- Run `python download_models.py --list` to verify

---

## Cost Estimation

| GPU | Price/hr | ~Time/video | ~Cost/video |
|-----|----------|-------------|-------------|
| A100 80GB | $1.99 | 30s | ~$0.02 |
| A10G 24GB | $0.44 | 60s | ~$0.007 |
| H100 80GB | $3.99 | 20s | ~$0.02 |

---

## Links

- [LTX-2 GitHub](https://github.com/Lightricks/LTX-2)
- [LTX-2 HuggingFace](https://huggingface.co/Lightricks/LTX-2)
- [RunPod Documentation](https://docs.runpod.io/)
- [Prompting Guide](https://ltx.video/blog/how-to-prompt-for-ltx-2)
