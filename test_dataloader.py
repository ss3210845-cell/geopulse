from flood_dataset import FloodDataset
from torch.utils.data import DataLoader


dataset = FloodDataset(
    csv_file="data/raw/sen1floods11/metadata/train_data.csv",
    base_dir="data/raw/sen1floods11",
    num_samples=20
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

images, flood_masks, valid_masks = next(iter(loader))

print("DataLoader Test")
print("=" * 50)

print("Images shape:", images.shape)
print("Flood masks shape:", flood_masks.shape)
print("Valid masks shape:", valid_masks.shape)

print()
print("Image dtype:", images.dtype)
print("Flood mask dtype:", flood_masks.dtype)
print("Valid mask dtype:", valid_masks.dtype)

print()
print("Batch flood pixels:", flood_masks.sum().item())
print("Batch usable pixels:", valid_masks.sum().item())

print()
print("DataLoader test complete!")