import os
os.environ['KERAS_BACKEND'] = 'torch'

print("=" * 50)

# 1. PyTorch
import torch
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA ok  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU      : NAO ENCONTRADA — verifica --gpus all no docker run")

print()

# 2. Keras 3
import keras
print(f"Keras    : {keras.__version__}")
print(f"Backend  : {keras.backend.backend()}")

print()

# 3. Teste rapido: tensor na GPU
if torch.cuda.is_available():
    x = torch.randn(100, 100, device='cuda')
    y = torch.matmul(x, x)
    print(f"Teste GPU: OK (matmul 100x100 na {torch.cuda.get_device_name(0)})")

print("=" * 50)
