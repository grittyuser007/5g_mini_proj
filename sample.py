import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")