#!/usr/bin/env python3
"""
Model download script for vLLM-omni
Downloads commonly used AR and DiT models
"""

import os
import sys
import argparse
from pathlib import Path

def download_ar_models(models=None):
    """Download AR (Autoregressive) models"""
    if models is None:
        models = [
            "Qwen/Qwen3-0.6B",
        ]
    
    print("🚀 Downloading AR models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("❌ transformers not installed. Installing...")
        os.system("pip install transformers")
        from transformers import AutoTokenizer, AutoModel
    
    for model_name in models:
        print(f"\n📥 Downloading AR model: {model_name}")
        try:
            print(f"  - Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"  - Downloading model...")
            model = AutoModel.from_pretrained(model_name)
            
            print(f"✅ {model_name} downloaded successfully")
            
            # Print model info
            total_params = sum(p.numel() for p in model.parameters())
            size_mb = total_params * 4 / (1024 * 1024)
            print(f"   📊 Parameters: {total_params:,} ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def download_dit_models(models=None):
    """Download DiT (Diffusion Transformer) models"""
    if models is None:
        models = [
            "stabilityai/stable-diffusion-2-1",
            "runwayml/stable-diffusion-v1-5"
        ]
    
    print("\n🚀 Downloading DiT models...")
    
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("❌ diffusers not installed. Installing...")
        os.system("pip install diffusers")
        from diffusers import StableDiffusionPipeline
    
    for model_name in models:
        print(f"\n📥 Downloading DiT model: {model_name}")
        try:
            print(f"  - Downloading pipeline...")
            pipe = StableDiffusionPipeline.from_pretrained(model_name)
            
            print(f"✅ {model_name} downloaded successfully")
            
            # Print model info
            total_params = sum(p.numel() for p in pipe.unet.parameters())
            size_mb = total_params * 4 / (1024 * 1024)
            print(f"   📊 UNet parameters: {total_params:,} ({size_mb:.1f} MB)")
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

def check_cache_directory():
    """Check and display cache directory info"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    print(f"📁 Cache directory: {cache_dir}")
    
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        print(f"📊 Total cache size: {size_gb:.2f} GB")
        
        # List downloaded models
        model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
        if model_dirs:
            print(f"📋 Downloaded models ({len(model_dirs)}):")
            for model_dir in sorted(model_dirs):
                model_name = model_dir.name.replace("models--", "").replace("--", "/")
                print(f"   - {model_name}")
    else:
        print("📁 Cache directory does not exist yet")

def clear_cache():
    """Clear model cache"""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if cache_dir.exists():
        print(f"🗑️  Clearing cache directory: {cache_dir}")
        import shutil
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared successfully")
    else:
        print("📁 Cache directory does not exist")

def main():
    parser = argparse.ArgumentParser(description="Download models for vLLM-omni")
    parser.add_argument("--ar-models", nargs="+", help="AR models to download")
    parser.add_argument("--dit-models", nargs="+", help="DiT models to download")
    parser.add_argument("--all", action="store_true", help="Download all default models")
    parser.add_argument("--check-cache", action="store_true", help="Check cache directory")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache directory")
    
    args = parser.parse_args()
    
    print("🎯 vLLM-omni Model Downloader")
    print("=" * 40)
    
    if args.clear_cache:
        clear_cache()
        return
    
    if args.check_cache:
        check_cache_directory()
        return
    
    if args.all or args.ar_models:
        download_ar_models(args.ar_models)
    
    if args.all or args.dit_models:
        download_dit_models(args.dit_models)
    
    if not any([args.all, args.ar_models, args.dit_models, args.check_cache, args.clear_cache]):
        print("❓ No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python scripts/download_models.py --all")
        print("  python scripts/download_models.py --ar-models Qwen/Qwen3-0.6B")
        print("  python scripts/download_models.py --check-cache")

if __name__ == "__main__":
    main()
