#!/usr/bin/env python3
"""
LTX-2 Model Download Script for RunPod
Downloads all required models from HuggingFace to the specified directory.

Usage:
    # Download essential models only (for basic T2V/I2V)
    python download_models.py --model-dir ./models
    
    # Download all models including all LoRAs
    python download_models.py --model-dir ./models --all
    
    # Download specific LoRAs
    python download_models.py --model-dir ./models --loras camera-control detailer
    
Environment Variables:
    MODEL_DIR: Directory to store models (default: /runpod-volume/models)
    HF_TOKEN: HuggingFace token for gated models (optional)
"""

import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

# Main LTX-2 checkpoints (choose one)
MAIN_CHECKPOINTS = {
    "distilled-fp8": {
        "repo_id": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-distilled-fp8.safetensors",
        "description": "Distilled FP8 (fastest, lowest memory)",
        "size_gb": 19,
    },
    "distilled": {
        "repo_id": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-distilled.safetensors",
        "description": "Distilled BF16 (fast)",
        "size_gb": 38,
    },
    "dev-fp8": {
        "repo_id": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-dev-fp8.safetensors",
        "description": "Dev FP8 (best quality, low memory)",
        "size_gb": 19,
    },
    "dev": {
        "repo_id": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-dev.safetensors",
        "description": "Dev BF16 (best quality)",
        "size_gb": 38,
    },
}

# Required components
REQUIRED_MODELS = [
    {
        "repo_id": "Lightricks/LTX-2",
        "filename": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "description": "Spatial Upscaler 2x (required for two-stage pipelines)",
        "size_mb": 500,
    },
    {
        "repo_id": "Lightricks/LTX-2",
        "filename": "ltx-2-19b-distilled-lora-384.safetensors",
        "description": "Distilled LoRA (required for TI2VidTwoStagesPipeline)",
        "size_mb": 150,
    },
]

# Optional: Temporal upscaler
TEMPORAL_UPSCALER = {
    "repo_id": "Lightricks/LTX-2",
    "filename": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
    "description": "Temporal Upscaler 2x (for future pipeline features)",
    "size_mb": 500,
}

# IC-LoRAs (for video-to-video transformations)
IC_LORAS = {
    "canny": {
        "repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control",
        "filename": "ltx-2-19b-ic-lora-canny-control.safetensors",
        "description": "Canny Edge Control",
    },
    "depth": {
        "repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Depth-Control",
        "filename": "ltx-2-19b-ic-lora-depth-control.safetensors",
        "description": "Depth Control",
    },
    "detailer": {
        "repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Detailer",
        "filename": "ltx-2-19b-ic-lora-detailer.safetensors",
        "description": "Detailer (upscale/enhance)",
    },
    "pose": {
        "repo_id": "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control",
        "filename": "ltx-2-19b-ic-lora-pose-control.safetensors",
        "description": "Pose Control",
    },
}

# Camera Control LoRAs
CAMERA_LORAS = {
    "dolly-in": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In",
        "filename": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        "description": "Camera Dolly In",
    },
    "dolly-out": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out",
        "filename": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        "description": "Camera Dolly Out",
    },
    "dolly-left": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left",
        "filename": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        "description": "Camera Dolly Left",
    },
    "dolly-right": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right",
        "filename": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        "description": "Camera Dolly Right",
    },
    "jib-up": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up",
        "filename": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        "description": "Camera Jib Up",
    },
    "jib-down": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down",
        "filename": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        "description": "Camera Jib Down",
    },
    "static": {
        "repo_id": "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static",
        "filename": "ltx-2-19b-lora-camera-control-static.safetensors",
        "description": "Camera Static (locked)",
    },
}

# Gemma text encoder
GEMMA_MODEL = {
    "repo_id": "google/gemma-3-12b-it-qat-q4_0-unquantized",
    "description": "Gemma 3 Text Encoder (required)",
    "size_gb": 12,
}


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_file(repo_id: str, filename: str, output_dir: Path, token: str = None) -> Path:
    """Download a single file from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download
    
    output_path = output_dir / filename
    
    if output_path.exists():
        logger.info(f"  ✓ Already exists: {filename}")
        return output_path
    
    logger.info(f"  ↓ Downloading: {filename}")
    
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    
    return output_path


def download_gemma(output_dir: Path, token: str = None) -> Path:
    """Download the complete Gemma text encoder."""
    from huggingface_hub import snapshot_download
    
    gemma_path = output_dir / "gemma-3-12b-it-qat-q4_0-unquantized"
    
    if gemma_path.exists() and any(gemma_path.iterdir()):
        logger.info(f"  ✓ Already exists: Gemma 3 Text Encoder")
        return gemma_path
    
    logger.info(f"  ↓ Downloading: Gemma 3 Text Encoder (~12GB)")
    
    snapshot_download(
        repo_id=GEMMA_MODEL["repo_id"],
        local_dir=str(gemma_path),
        local_dir_use_symlinks=False,
        token=token,
    )
    
    return gemma_path


def get_dir_size(path: Path) -> float:
    """Get directory size in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)


def print_model_overview():
    """Print overview of all available models."""
    print("\n" + "=" * 70)
    print("LTX-2 MODEL OVERVIEW")
    print("=" * 70)
    
    print("\n📦 MAIN CHECKPOINTS (choose one):")
    for key, info in MAIN_CHECKPOINTS.items():
        print(f"  • {key}: {info['description']} (~{info['size_gb']}GB)")
    
    print("\n🔧 REQUIRED COMPONENTS:")
    for model in REQUIRED_MODELS:
        print(f"  • {model['filename']}: {model['description']}")
    print(f"  • Gemma 3 Text Encoder (~{GEMMA_MODEL['size_gb']}GB)")
    
    print("\n🎨 IC-LoRAs (video-to-video control):")
    for key, info in IC_LORAS.items():
        print(f"  • {key}: {info['description']}")
    
    print("\n🎬 CAMERA CONTROL LoRAs:")
    for key, info in CAMERA_LORAS.items():
        print(f"  • {key}: {info['description']}")
    
    print("\n📹 OPTIONAL:")
    print(f"  • Temporal Upscaler: {TEMPORAL_UPSCALER['description']}")
    
    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download LTX-2 models for RunPod deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Essential models only (T2V/I2V with DistilledPipeline)
  python download_models.py --model-dir ./models
  
  # All models including all LoRAs  
  python download_models.py --model-dir ./models --all
  
  # Custom selection
  python download_models.py --model-dir ./models --variant dev-fp8 --ic-loras canny depth --camera-loras
  
  # Show available models
  python download_models.py --list
        """
    )
    
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "./models"),
        help="Directory to store models (default: ./models)"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token (set HF_TOKEN env var or use --hf-token)"
    )
    parser.add_argument(
        "--variant",
        choices=list(MAIN_CHECKPOINTS.keys()),
        default="distilled-fp8",
        help="Main checkpoint variant (default: distilled-fp8)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download ALL models including all LoRAs"
    )
    parser.add_argument(
        "--ic-loras",
        nargs="*",
        choices=list(IC_LORAS.keys()) + ["all"],
        help="IC-LoRAs to download (canny, depth, detailer, pose, or 'all')"
    )
    parser.add_argument(
        "--camera-loras",
        nargs="*",
        choices=list(CAMERA_LORAS.keys()) + ["all"],
        default=None,
        help="Camera LoRAs to download (or just --camera-loras for all)"
    )
    parser.add_argument(
        "--temporal-upscaler",
        action="store_true",
        help="Also download temporal upscaler"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit"
    )
    parser.add_argument(
        "--skip-gemma",
        action="store_true",
        help="Skip Gemma download (if already have it)"
    )
    
    args = parser.parse_args()
    
    # Show model list
    if args.list:
        print_model_overview()
        return
    
    # Install huggingface_hub if needed
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.info("Installing huggingface_hub...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download, snapshot_download
    
    # Create directories
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = model_dir / "loras"
    lora_dir.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("LTX-2 MODEL DOWNLOADER")
    print("=" * 70)
    print(f"\nModel directory: {model_dir.absolute()}")
    print(f"Checkpoint variant: {args.variant}")
    
    downloaded = []
    failed = []
    
    # 1. Download main checkpoint
    print("\n📦 MAIN CHECKPOINT")
    checkpoint = MAIN_CHECKPOINTS[args.variant]
    try:
        download_file(checkpoint["repo_id"], checkpoint["filename"], model_dir, args.hf_token)
        downloaded.append(checkpoint["filename"])
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        failed.append(checkpoint["filename"])
    
    # 2. Download required components
    print("\n🔧 REQUIRED COMPONENTS")
    for model in REQUIRED_MODELS:
        try:
            download_file(model["repo_id"], model["filename"], model_dir, args.hf_token)
            downloaded.append(model["filename"])
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed.append(model["filename"])
    
    # 3. Download Gemma
    if not args.skip_gemma:
        print("\n🧠 TEXT ENCODER")
        try:
            download_gemma(model_dir, args.hf_token)
            downloaded.append("gemma-3-12b-it-qat-q4_0-unquantized")
        except Exception as e:
            logger.error(f"  ✗ Failed to download Gemma: {e}")
            failed.append("gemma-3-12b-it-qat-q4_0-unquantized")
    
    # 4. Download IC-LoRAs
    ic_loras_to_download = []
    if args.all or (args.ic_loras and "all" in args.ic_loras):
        ic_loras_to_download = list(IC_LORAS.keys())
    elif args.ic_loras:
        ic_loras_to_download = [l for l in args.ic_loras if l != "all"]
    
    if ic_loras_to_download:
        print("\n🎨 IC-LoRAs")
        for lora_key in ic_loras_to_download:
            lora = IC_LORAS[lora_key]
            try:
                download_file(lora["repo_id"], lora["filename"], lora_dir, args.hf_token)
                downloaded.append(f"loras/{lora['filename']}")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                failed.append(lora["filename"])
    
    # 5. Download Camera LoRAs
    camera_loras_to_download = []
    if args.all:
        camera_loras_to_download = list(CAMERA_LORAS.keys())
    elif args.camera_loras is not None:
        if len(args.camera_loras) == 0 or "all" in args.camera_loras:
            camera_loras_to_download = list(CAMERA_LORAS.keys())
        else:
            camera_loras_to_download = args.camera_loras
    
    if camera_loras_to_download:
        print("\n🎬 CAMERA LoRAs")
        for lora_key in camera_loras_to_download:
            lora = CAMERA_LORAS[lora_key]
            try:
                download_file(lora["repo_id"], lora["filename"], lora_dir, args.hf_token)
                downloaded.append(f"loras/{lora['filename']}")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")
                failed.append(lora["filename"])
    
    # 6. Download Temporal Upscaler
    if args.all or args.temporal_upscaler:
        print("\n📹 TEMPORAL UPSCALER")
        try:
            download_file(
                TEMPORAL_UPSCALER["repo_id"],
                TEMPORAL_UPSCALER["filename"],
                model_dir,
                args.hf_token
            )
            downloaded.append(TEMPORAL_UPSCALER["filename"])
        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed.append(TEMPORAL_UPSCALER["filename"])
    
    # Summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    
    total_size = 0
    print("\n✓ Downloaded files:")
    for item in model_dir.iterdir():
        size = get_dir_size(item)
        total_size += size
        if size > 1024:
            print(f"  • {item.name}: {size/1024:.1f} GB")
        else:
            print(f"  • {item.name}: {size:.1f} MB")
    
    if lora_dir.exists() and any(lora_dir.iterdir()):
        print("\n  loras/")
        for item in lora_dir.iterdir():
            size = get_dir_size(item)
            total_size += size
            print(f"    • {item.name}: {size:.1f} MB")
    
    print(f"\nTotal size: {total_size/1024:.1f} GB")
    
    if failed:
        print(f"\n✗ Failed downloads ({len(failed)}):")
        for f in failed:
            print(f"  • {f}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
1. Build Docker image:
   docker build -t ltx2-runpod -f runpod/Dockerfile .

2. Push to registry:
   docker tag ltx2-runpod yourusername/ltx2-runpod:latest
   docker push yourusername/ltx2-runpod:latest

3. Create RunPod endpoint with this image and mount volume with models
    """)


if __name__ == "__main__":
    main()
