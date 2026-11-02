import sys
import platform
import torch

print("Python        :", sys.version.split()[0])
print("Platform      :", platform.platform())
print("PyTorch       :", torch.__version__)
print("MPS built     :", torch.backends.mps.is_built())
print("MPS available :", torch.backends.mps.is_available())

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device  :", device)

# GPU sanity test
x = torch.randn(2, 3, 256, 256, device=device)

conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)

output = conv(x)

print("Conv output shape:", tuple(output.shape))
