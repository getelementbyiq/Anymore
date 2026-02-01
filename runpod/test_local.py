#!/usr/bin/env python3
"""
Local test script for LTX-2 RunPod handler.
Run this before deploying to verify everything works.

Usage:
    python test_local.py
"""

import os
import sys
import base64
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set environment variables for local testing
os.environ.setdefault("MODEL_DIR", str(Path(__file__).parent / "models"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def test_text_to_video():
    """Test basic text-to-video generation."""
    print("=" * 60)
    print("Test 1: Text-to-Video Generation")
    print("=" * 60)
    
    from handler import handler
    
    job = {
        "input": {
            "prompt": "A golden retriever running through a sunny meadow, slow motion, cinematic lighting",
            "negative_prompt": "blurry, low quality, distorted",
            "seed": 42,
            "height": 544,
            "width": 960,
            "num_frames": 25,  # Shorter for faster testing
            "frame_rate": 25.0,
            "num_inference_steps": 8,
        }
    }
    
    print(f"Prompt: {job['input']['prompt']}")
    print(f"Resolution: {job['input']['width']}x{job['input']['height']}")
    print(f"Frames: {job['input']['num_frames']}")
    print("\nGenerating video...")
    
    result = handler(job)
    
    if result.get("status") == "success":
        print(f"✓ Success! Video generated")
        print(f"  Seed used: {result.get('seed')}")
        
        # Save video
        output_path = Path(__file__).parent / "outputs" / "test_t2v.mp4"
        output_path.parent.mkdir(exist_ok=True)
        
        video_bytes = base64.b64decode(result["video_base64"])
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        print(f"  Saved to: {output_path}")
        print(f"  File size: {len(video_bytes) / 1024 / 1024:.2f} MB")
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        return False
    
    return True


def test_image_to_video():
    """Test image-to-video generation."""
    print("\n" + "=" * 60)
    print("Test 2: Image-to-Video Generation")
    print("=" * 60)
    
    # Check if test image exists
    test_image_path = Path(__file__).parent / "test_image.png"
    if not test_image_path.exists():
        print(f"⚠ Skipping I2V test: No test image found at {test_image_path}")
        print("  Create a test_image.png in the runpod folder to run this test")
        return True
    
    from handler import handler
    
    # Load and encode test image
    with open(test_image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    job = {
        "input": {
            "prompt": "The scene comes to life with gentle movement",
            "seed": 42,
            "height": 544,
            "width": 960,
            "num_frames": 25,
            "images": [
                {
                    "base64": image_base64,
                    "frame_index": 0,
                    "strength": 1.0
                }
            ]
        }
    }
    
    print(f"Input image: {test_image_path}")
    print(f"Prompt: {job['input']['prompt']}")
    print("\nGenerating video...")
    
    result = handler(job)
    
    if result.get("status") == "success":
        print(f"✓ Success! Video generated")
        
        output_path = Path(__file__).parent / "outputs" / "test_i2v.mp4"
        video_bytes = base64.b64decode(result["video_base64"])
        with open(output_path, "wb") as f:
            f.write(video_bytes)
        print(f"  Saved to: {output_path}")
    else:
        print(f"✗ Failed: {result.get('error', 'Unknown error')}")
        return False
    
    return True


def check_models():
    """Check if all required models are present."""
    print("=" * 60)
    print("Checking Models")
    print("=" * 60)
    
    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    
    required_models = [
        ("ltx-2-19b-distilled-fp8.safetensors", "Main checkpoint"),
        ("ltx-2-spatial-upscaler-x2-1.0.safetensors", "Spatial upsampler"),
        ("ltx-2-19b-distilled-lora-384.safetensors", "Distilled LoRA"),
        ("gemma-3-12b-it-qat-q4_0-unquantized", "Gemma text encoder"),
    ]
    
    all_present = True
    for filename, description in required_models:
        path = model_dir / filename
        if path.exists():
            if path.is_file():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✓ {description}: {size_mb:.1f} MB")
            else:
                size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
                print(f"✓ {description}: {size_mb:.1f} MB (directory)")
        else:
            print(f"✗ {description}: NOT FOUND at {path}")
            all_present = False
    
    if not all_present:
        print("\n⚠ Some models are missing!")
        print(f"  Run: python download_models.py --model-dir {model_dir}")
    
    return all_present


def main():
    print("\nLTX-2 RunPod Local Test")
    print("=" * 60)
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            print("⚠ CUDA not available - tests will fail")
            return
    except ImportError:
        print("✗ PyTorch not installed")
        return
    
    print()
    
    # Check models
    if not check_models():
        print("\n✗ Cannot run tests without models")
        return
    
    print()
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Test 1: T2V
    tests_total += 1
    if test_text_to_video():
        tests_passed += 1
    
    # Test 2: I2V
    tests_total += 1
    if test_image_to_video():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("✓ All tests passed! Ready for deployment.")
    else:
        print("⚠ Some tests failed. Check the output above.")


if __name__ == "__main__":
    main()
