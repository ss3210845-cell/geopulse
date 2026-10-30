import torch

from unet import UNet


print("Testing GeoPulse U-Net")
print("=" * 50)

# Create model
model = UNet()

print("Model created successfully!")

# Create fake batch
x = torch.randn(
    4,
    2,
    512,
    512
)

print("Input shape:", x.shape)

# Forward pass
with torch.no_grad():
    output = model(x)

print("Output shape:", output.shape)

print()
print("U-Net test complete!")