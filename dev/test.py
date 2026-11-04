import torch
import torch.nn as nn

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("cuDNN available:", torch.backends.cudnn.is_available())
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("GPU:", torch.cuda.get_device_name(0))

device = "cuda"

# Test 1: Matrix multiplication
print("\n[1] Matrix multiplication", flush=True)

a = torch.randn(512, 512, device=device)
b = torch.randn(512, 512, device=device)

c = a @ b
torch.cuda.synchronize()

print("[1] OK", flush=True)

# Test 2: Convolution with cuDNN enabled
print("\n[2] Convolution with cuDNN enabled", flush=True)

torch.backends.cudnn.enabled = True

x = torch.randn(1, 3, 256, 256, device=device)
conv = nn.Conv2d(3, 64, 3, padding=1).to(device)

y = conv(x)
torch.cuda.synchronize()

print("[2] OK:", y.shape, flush=True)

# Test 3: Convolution with cuDNN disabled
print("\n[3] Convolution with cuDNN disabled", flush=True)

torch.backends.cudnn.enabled = False

y = conv(x)
torch.cuda.synchronize()

print("[3] OK:", y.shape, flush=True)

print("\nALL TESTS PASSED", flush=True)