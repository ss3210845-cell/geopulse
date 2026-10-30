import torch

from unet import UNet
from loss_functions import CombinedLoss


print("Testing GeoPulse Loss Function")
print("=" * 50)

# Create model
model = UNet()

# Create fake input
images = torch.randn(
    2,
    2,
    512,
    512
)

# Fake flood masks
targets = torch.randint(
    0,
    2,
    (2, 512, 512)
).float()

# All pixels valid for this test
valid_mask = torch.ones(
    2,
    512,
    512
)

# Model prediction
logits = model(images)

print("Prediction shape:", logits.shape)

# Create loss
criterion = CombinedLoss()

# Calculate loss
loss = criterion(
    logits,
    targets,
    valid_mask
)

print("Loss:", loss.item())

print()
print("Loss test complete!")