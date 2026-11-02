"""Phase 3: Verify U-Net architecture and gradient flow."""

import time
import yaml
import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import build_model, count_parameters
from src.utils import get_device, set_seed


with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["seed"])

device = get_device()

print("Device:", device)

# Build U-Net
model = build_model(cfg).to(device)

print(
    "Trainable parameters: %.2fM"
    % (count_parameters(model) / 1e6)
)

# 1. Inspect layer output shapes
shapes = []

hooks = []

for name, module in model.named_children():

    hook = module.register_forward_hook(
        lambda mod, inp, out, n=name:
        shapes.append((n, tuple(out.shape)))
    )

    hooks.append(hook)

model.eval()

with torch.no_grad():

    sample = torch.randn(
        2, 3, 256, 256, device=device
    )

    out = model(sample)

for hook in hooks:
    hook.remove()

print("\nArchitecture shape trace:")

for name, shape in shapes:
    print(f"{name:6s} -> {shape}")

assert out.shape == (2, 1, 256, 256)

print(
    "\nOutput logits range: [%.2f, %.2f]"
    % (out.min().item(), out.max().item())
)

# 2. Real batch overfit test
print("\nStarting one-batch overfit test...")

train_dl, _, _ = get_dataloaders()

x, y = next(iter(train_dl))

x = x.to(device)
y = y.to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=1e-3
)

model.train()

start = time.time()

for step in range(1, 41):

    optimizer.zero_grad()

    logits = model(x)

    loss = criterion(logits, y)

    assert not torch.isnan(loss), "NaN loss detected!"

    loss.backward()

    optimizer.step()

    if step in (1, 10, 20, 30, 40):

        print(
            f"Step {step:2d} | Loss: {loss.item():.4f}"
        )

if device.type == "mps":
    torch.mps.synchronize()

print(
    "\n40 steps completed in %.1f seconds"
    % (time.time() - start)
)

print("U-Net verification completed successfully!")
