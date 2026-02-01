"""
LTX-2 RunPod Serverless Handler
Handles video generation requests via RunPod's serverless infrastructure.

Supports all pipelines:
- DistilledPipeline (default, fastest)
- TI2VidTwoStagesPipeline (best quality)
- TI2VidOneStagePipeline (quick prototyping)
- ICLoraPipeline (video-to-video with control LoRAs)
- KeyframeInterpolationPipeline (interpolate between images)
"""

import os
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional

import torch
import runpod

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/runpod-volume/models"))
LORA_DIR = MODEL_DIR / "loras"
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/huggingface")

# Model paths - auto-detect checkpoint variant
def find_checkpoint() -> str:
    """Find the main checkpoint file."""
    variants = [
        "ltx-2-19b-distilled-fp8.safetensors",
        "ltx-2-19b-distilled.safetensors",
        "ltx-2-19b-dev-fp8.safetensors",
        "ltx-2-19b-dev.safetensors",
    ]
    for variant in variants:
        path = MODEL_DIR / variant
        if path.exists():
            logger.info(f"Found checkpoint: {variant}")
            return str(path)
    raise FileNotFoundError(f"No checkpoint found in {MODEL_DIR}")

# Required model paths
UPSAMPLER_PATH = MODEL_DIR / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA_PATH = MODEL_DIR / "ltx-2-19b-distilled-lora-384.safetensors"
GEMMA_PATH = MODEL_DIR / "gemma-3-12b-it-qat-q4_0-unquantized"

# Available LoRAs
AVAILABLE_LORAS = {
    # IC-LoRAs
    "canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
    "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
    "pose": "ltx-2-19b-ic-lora-pose-control.safetensors",
    # Camera LoRAs
    "dolly-in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly-out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly-left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly-right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "jib-up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib-down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "static": "ltx-2-19b-lora-camera-control-static.safetensors",
}

# Global pipeline cache
pipelines = {}


# =============================================================================
# PIPELINE LOADING
# =============================================================================

def get_lora_path(lora_name: str) -> Optional[str]:
    """Get path to a LoRA file if it exists."""
    if lora_name not in AVAILABLE_LORAS:
        return None
    path = LORA_DIR / AVAILABLE_LORAS[lora_name]
    return str(path) if path.exists() else None


def load_pipeline(pipeline_type: str = "distilled", loras: list = None, fp8: bool = True):
    """
    Load or retrieve cached pipeline.
    
    Args:
        pipeline_type: One of 'distilled', 'two_stage', 'one_stage', 'ic_lora', 'keyframe'
        loras: List of LoRA names to apply
        fp8: Whether to use FP8 precision
    """
    global pipelines
    
    # Create cache key
    lora_key = tuple(sorted(loras)) if loras else ()
    cache_key = (pipeline_type, lora_key, fp8)
    
    if cache_key in pipelines:
        logger.info(f"Using cached pipeline: {pipeline_type}")
        return pipelines[cache_key]
    
    logger.info(f"Loading pipeline: {pipeline_type} (fp8={fp8})")
    
    # Find checkpoint
    checkpoint_path = find_checkpoint()
    is_fp8_checkpoint = "fp8" in checkpoint_path
    
    # Verify required files
    if not GEMMA_PATH.exists():
        raise FileNotFoundError(f"Gemma not found at {GEMMA_PATH}")
    
    # Import components
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    
    # Build LoRA list
    lora_configs = []
    if loras:
        for lora_name in loras:
            lora_path = get_lora_path(lora_name)
            if lora_path:
                lora_configs.append(
                    LoraPathStrengthAndSDOps(lora_path, 0.8, LTXV_LORA_COMFY_RENAMING_MAP)
                )
                logger.info(f"  + LoRA: {lora_name}")
            else:
                logger.warning(f"  ! LoRA not found: {lora_name}")
    
    # Build distilled LoRA config
    distilled_lora = []
    if DISTILLED_LORA_PATH.exists():
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                str(DISTILLED_LORA_PATH), 0.8, LTXV_LORA_COMFY_RENAMING_MAP
            )
        ]
    
    # Load appropriate pipeline
    if pipeline_type == "distilled":
        from ltx_pipelines.distilled import DistilledPipeline
        pipeline = DistilledPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=str(UPSAMPLER_PATH),
            gemma_root=str(GEMMA_PATH),
            loras=lora_configs,
            fp8transformer=fp8 and not is_fp8_checkpoint,
        )
        
    elif pipeline_type == "two_stage":
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=str(UPSAMPLER_PATH),
            gemma_root=str(GEMMA_PATH),
            loras=lora_configs,
            fp8transformer=fp8 and not is_fp8_checkpoint,
        )
        
    elif pipeline_type == "one_stage":
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=str(GEMMA_PATH),
            loras=lora_configs,
            fp8transformer=fp8 and not is_fp8_checkpoint,
        )
        
    elif pipeline_type == "ic_lora":
        from ltx_pipelines.ic_lora import ICLoraPipeline
        # IC-LoRA requires an IC-LoRA to be specified
        if not lora_configs:
            raise ValueError("IC-LoRA pipeline requires at least one IC-LoRA (canny, depth, detailer, pose)")
        pipeline = ICLoraPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=str(UPSAMPLER_PATH),
            gemma_root=str(GEMMA_PATH),
            ic_lora=lora_configs[0],  # Primary IC-LoRA
            loras=lora_configs[1:],   # Additional LoRAs
            fp8transformer=fp8 and not is_fp8_checkpoint,
        )
        
    elif pipeline_type == "keyframe":
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=str(UPSAMPLER_PATH),
            gemma_root=str(GEMMA_PATH),
            loras=lora_configs,
            fp8transformer=fp8 and not is_fp8_checkpoint,
        )
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    # Cache and return
    pipelines[cache_key] = pipeline
    logger.info(f"Pipeline loaded successfully!")
    return pipeline


def generate_video(job_input: dict) -> dict:
    """
    Generate a video from the given input parameters.
    
    Args:
        job_input: Dictionary containing:
            - prompt (str): Text description of the video
            - negative_prompt (str, optional): What to avoid
            - seed (int, optional): Random seed for reproducibility
            - height (int, optional): Output height (default: 544)
            - width (int, optional): Output width (default: 960)
            - num_frames (int, optional): Number of frames (default: 49)
            - frame_rate (float, optional): FPS (default: 25.0)
            - num_inference_steps (int, optional): Denoising steps (default: 8 for distilled, 40 for others)
            - images (list, optional): List of input images for I2V
            - enhance_prompt (bool, optional): Auto-enhance prompt
            - pipeline (str, optional): Pipeline type - 'distilled', 'two_stage', 'one_stage', 'ic_lora', 'keyframe'
            - loras (list, optional): List of LoRA names to apply
            - fp8 (bool, optional): Use FP8 precision (default: True)
            - guidance (dict, optional): Guidance parameters (cfg_scale, stg_scale, etc.)
            - reference_video (str, optional): Base64 video for IC-LoRA V2V
            
    Returns:
        Dictionary with:
            - video_base64: Base64-encoded MP4 video
            - seed: Used seed
            - prompt: Used prompt (possibly enhanced)
            - pipeline: Pipeline used
            - status: 'success' or error info
    """
    # Extract parameters
    prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
    negative_prompt = job_input.get("negative_prompt", "")
    seed = job_input.get("seed", 42)
    height = job_input.get("height", 544)
    width = job_input.get("width", 960)
    num_frames = job_input.get("num_frames", 49)
    frame_rate = job_input.get("frame_rate", 25.0)
    enhance_prompt = job_input.get("enhance_prompt", False)
    
    # Pipeline configuration
    pipeline_type = job_input.get("pipeline", "distilled")
    loras = job_input.get("loras", [])
    fp8 = job_input.get("fp8", True)
    
    # Set default steps based on pipeline
    default_steps = 8 if pipeline_type == "distilled" else 40
    num_inference_steps = job_input.get("num_inference_steps", default_steps)
    
    # Guidance parameters
    guidance = job_input.get("guidance", {})
    video_cfg_scale = guidance.get("video_cfg_scale", 3.0)
    video_stg_scale = guidance.get("video_stg_scale", 1.0)
    audio_cfg_scale = guidance.get("audio_cfg_scale", 7.0)
    audio_stg_scale = guidance.get("audio_stg_scale", 1.0)
    
    # Load pipeline
    pipeline = load_pipeline(pipeline_type, loras, fp8)
    
    # Handle image conditioning (I2V)
    images = []
    temp_files = []
    if "images" in job_input:
        for img_data in job_input["images"]:
            img_bytes = base64.b64decode(img_data["base64"])
            temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_img.write(img_bytes)
            temp_img.close()
            temp_files.append(temp_img.name)
            
            images.append((
                temp_img.name,
                img_data.get("frame_index", 0),
                img_data.get("strength", 1.0)
            ))
    
    # Handle reference video for IC-LoRA
    reference_video_path = None
    if "reference_video" in job_input and pipeline_type == "ic_lora":
        video_bytes = base64.b64decode(job_input["reference_video"])
        temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video.write(video_bytes)
        temp_video.close()
        reference_video_path = temp_video.name
        temp_files.append(reference_video_path)
    
    logger.info(f"Generating video with {pipeline_type} pipeline")
    logger.info(f"  Prompt: {prompt[:50]}...")
    logger.info(f"  Resolution: {width}x{height}, Frames: {num_frames}")
    logger.info(f"  LoRAs: {loras if loras else 'none'}")
    
    # Create temp output file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_output:
        output_path = temp_output.name
    
    try:
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_core.components.guiders import MultiModalGuiderParams
        
        tiling_config = TilingConfig.default()
        video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        
        # Build kwargs based on pipeline type
        kwargs = {
            "prompt": prompt,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "tiling_config": tiling_config,
        }
        
        # Add pipeline-specific parameters
        if pipeline_type in ["distilled"]:
            kwargs["negative_prompt"] = negative_prompt
            kwargs["images"] = images
            kwargs["enhance_prompt"] = enhance_prompt
            
        elif pipeline_type in ["two_stage", "one_stage"]:
            kwargs["negative_prompt"] = negative_prompt
            kwargs["num_inference_steps"] = num_inference_steps
            kwargs["images"] = images
            kwargs["enhance_prompt"] = enhance_prompt
            kwargs["video_guider_params"] = MultiModalGuiderParams(
                cfg_scale=video_cfg_scale,
                stg_scale=video_stg_scale,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[29],
            )
            kwargs["audio_guider_params"] = MultiModalGuiderParams(
                cfg_scale=audio_cfg_scale,
                stg_scale=audio_stg_scale,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[29],
            )
            
        elif pipeline_type == "ic_lora":
            kwargs["negative_prompt"] = negative_prompt
            kwargs["num_inference_steps"] = num_inference_steps
            kwargs["images"] = images
            if reference_video_path:
                kwargs["reference_video"] = reference_video_path
                
        elif pipeline_type == "keyframe":
            kwargs["negative_prompt"] = negative_prompt
            kwargs["num_inference_steps"] = num_inference_steps
            kwargs["keyframes"] = images  # For keyframe pipeline
        
        # Generate video
        video, audio = pipeline(**kwargs)
        
        # Encode to MP4
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        
        # Read and encode as base64
        with open(output_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        file_size_mb = len(video_base64) * 3 / 4 / (1024 * 1024)  # Approximate actual size
        logger.info(f"Video generated! Size: ~{file_size_mb:.1f} MB")
        
        return {
            "video_base64": video_base64,
            "seed": seed,
            "prompt": prompt,
            "pipeline": pipeline_type,
            "loras": loras,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise
        
    finally:
        # Cleanup temp files
        if os.path.exists(output_path):
            os.unlink(output_path)
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


def get_status() -> dict:
    """Get system status and available models."""
    status = {
        "gpu": None,
        "gpu_memory_gb": None,
        "checkpoint": None,
        "gemma": False,
        "upsampler": False,
        "distilled_lora": False,
        "available_loras": [],
        "available_pipelines": ["distilled", "two_stage", "one_stage", "ic_lora", "keyframe"],
    }
    
    # GPU info
    if torch.cuda.is_available():
        status["gpu"] = torch.cuda.get_device_name(0)
        status["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
    
    # Check models
    try:
        status["checkpoint"] = Path(find_checkpoint()).name
    except FileNotFoundError:
        pass
    
    status["gemma"] = GEMMA_PATH.exists()
    status["upsampler"] = UPSAMPLER_PATH.exists()
    status["distilled_lora"] = DISTILLED_LORA_PATH.exists()
    
    # Check available LoRAs
    for lora_name, lora_file in AVAILABLE_LORAS.items():
        if (LORA_DIR / lora_file).exists():
            status["available_loras"].append(lora_name)
    
    return status


def handler(job: dict) -> dict:
    """
    RunPod handler function.
    
    Args:
        job: RunPod job dictionary containing 'input' key
        
    Input actions:
        - {"action": "status"} - Get system status and available models
        - {"action": "generate", ...} - Generate video (default action)
        - {...} - Generate video (backwards compatible)
        
    Returns:
        Result dictionary or error
    """
    try:
        job_input = job.get("input", {})
        
        if not job_input:
            return {"error": "No input provided"}
        
        # Handle different actions
        action = job_input.get("action", "generate")
        
        if action == "status":
            return get_status()
        
        elif action == "generate" or "prompt" in job_input:
            result = generate_video(job_input)
            return result
        
        else:
            return {"error": f"Unknown action: {action}"}
        
    except FileNotFoundError as e:
        return {"error": str(e), "status": "model_not_found"}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "error": "Out of GPU memory. Try reducing resolution, num_frames, or use fp8=True",
            "status": "oom"
        }
    except Exception as e:
        logger.exception("Unexpected error in handler")
        return {"error": str(e), "status": "error"}


# =============================================================================
# RUNPOD ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Local testing mode
    if "--test" in sys.argv:
        print("\n" + "=" * 60)
        print("LTX-2 LOCAL TEST")
        print("=" * 60)
        
        # Check status first
        print("\n📊 System Status:")
        status = get_status()
        print(f"  GPU: {status['gpu']} ({status['gpu_memory_gb']} GB)")
        print(f"  Checkpoint: {status['checkpoint']}")
        print(f"  Gemma: {'✓' if status['gemma'] else '✗'}")
        print(f"  Upsampler: {'✓' if status['upsampler'] else '✗'}")
        print(f"  Available LoRAs: {status['available_loras'] or 'none'}")
        
        if not status["checkpoint"]:
            print("\n✗ No checkpoint found. Run download_models.py first.")
            sys.exit(1)
        
        # Test generation
        print("\n🎬 Testing video generation...")
        test_input = {
            "input": {
                "prompt": "A golden retriever running through a sunny meadow, slow motion",
                "seed": 42,
                "height": 544,
                "width": 960,
                "num_frames": 25,  # Shorter for testing
                "pipeline": "distilled",
            }
        }
        
        result = handler(test_input)
        
        if result.get("status") == "success":
            print(f"✓ Success!")
            print(f"  Seed: {result.get('seed')}")
            print(f"  Pipeline: {result.get('pipeline')}")
            
            # Save video
            video_bytes = base64.b64decode(result["video_base64"])
            output_path = Path("test_output.mp4")
            with open(output_path, "wb") as f:
                f.write(video_bytes)
            print(f"  Saved to: {output_path} ({len(video_bytes) / 1024 / 1024:.1f} MB)")
        else:
            print(f"✗ Failed: {result.get('error', 'Unknown error')}")
    
    else:
        # RunPod serverless mode
        logger.info("Starting LTX-2 RunPod handler...")
        runpod.serverless.start({"handler": handler})
