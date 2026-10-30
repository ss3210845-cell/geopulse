from flood_dataset import FloodDataset


dataset = FloodDataset(
    csv_file="data/raw/sen1floods11/metadata/train_data.csv",
    base_dir="data/raw/sen1floods11",
    num_samples=20
)

print("Dataset size:", len(dataset))

image, flood_mask, valid_mask = dataset[0]

print()
print("Image:")
print("Shape:", image.shape)
print("Type:", image.dtype)

print()
print("Flood mask:")
print("Shape:", flood_mask.shape)
print("Type:", flood_mask.dtype)

print()
print("Valid mask:")
print("Shape:", valid_mask.shape)
print("Type:", valid_mask.dtype)

print()
print("Image statistics:")
print("VV mean:", image[0].mean().item())
print("VV std:", image[0].std().item())

print()
print("Flood pixels:", flood_mask.sum().item())
print("Usable pixels:", valid_mask.sum().item())

print()
print("Dataset test complete!")