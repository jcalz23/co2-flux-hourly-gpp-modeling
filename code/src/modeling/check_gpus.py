import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {device_count}")
else:
    print("GPU is not available on this system.")
