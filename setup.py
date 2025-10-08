import os
import subprocess
import sys

def install_requirements():
    """安装所需依赖"""
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.6.0",
        "trl>=0.7.0",
        "datasets>=2.14.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.24.0",
        "scipy>=1.11.0",
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
    
    print("✓ All packages installed!")

def setup_directories():
    """创建必要的目录结构"""
    directories = [
        "/content/drive/MyDrive/absa-c-cos",
        "/content/drive/MyDrive/absa-c-cos/data",
        "/content/drive/MyDrive/absa-c-cos/models",
        "/content/drive/MyDrive/absa-c-cos/models/qwen_absa_v2",
        "/content/drive/MyDrive/absa-c-cos/logs",
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")

def verify_gpu():
    """验证GPU可用性"""
    import torch
    
    print("Checking GPU availability...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU: {gpu_name}")
        print(f"✓ Memory: {gpu_memory:.1f} GB")
        return True
    else:
        print("❌ No GPU available")
        return False

def main():
    print("=== ABSA-C-CoS v2.0 Setup ===")
    
    # 验证环境
    if not verify_gpu():
        print("Warning: Training may be very slow without GPU")
    
    # 安装依赖
    install_requirements()
    
    # 创建目录
    setup_directories()
    
    print("\n✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Upload your dataset files to /content/drive/MyDrive/absa-c-cos/data/")
    print("2. Run: python train.py")
    print("3. Run: python inference.py")

if __name__ == "__main__":
    main()