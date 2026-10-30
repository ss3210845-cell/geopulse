import torch
from torch.utils.data import DataLoader

from flood_dataset import FloodDataset
from unet import UNet
from loss_functions import CombinedLoss
from metrics import calculate_dice, calculate_iou


# ============================================================
# Configuration
# ============================================================

TOTAL_SAMPLES = 30

# Validation samples:
# 13, 20, 23, 25, 26, 30
VAL_INDICES = [12, 19, 22, 24, 25, 29]

# Sample 28 has zero valid pixels
EXCLUDED_INDEX = 27

# All remaining samples are used for training
TRAIN_INDICES = [
    i
    for i in range(TOTAL_SAMPLES)
    if i not in VAL_INDICES
    and i != EXCLUDED_INDEX
]

BATCH_SIZE = 1

EPOCHS = 20

LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

best_dice = 0.0
best_epoch = 0


# ============================================================
# Configuration information
# ============================================================

print("GeoPulse Flood Segmentation Training")
print("=" * 60)

print(f"Total samples       : {TOTAL_SAMPLES}")
print(f"Training samples    : {len(TRAIN_INDICES)}")
print(f"Validation samples  : {len(VAL_INDICES)}")
print(f"Excluded sample     : {EXCLUDED_INDEX + 1}")
print(f"Batch size          : {BATCH_SIZE}")
print(f"Epochs              : {EPOCHS}")
print(f"Learning rate       : {LEARNING_RATE}")
print(f"Device              : {DEVICE}")

print("\nTraining samples:")
print([i + 1 for i in TRAIN_INDICES])

print("\nValidation samples:")
print([i + 1 for i in VAL_INDICES])

print("=" * 60)


# ============================================================
# Dataset
# ============================================================

BASE_DIR = "data/raw/sen1floods11"

CSV_FILE = (
    "data/raw/sen1floods11/"
    "metadata/train_data.csv"
)

print("\nCreating training dataset...")

train_dataset = FloodDataset(
    csv_file=CSV_FILE,
    base_dir=BASE_DIR,
    num_samples=TOTAL_SAMPLES,
    indices=TRAIN_INDICES
)

print("Training dataset created.")


print("\nCreating validation dataset...")

val_dataset = FloodDataset(
    csv_file=CSV_FILE,
    base_dir=BASE_DIR,
    num_samples=TOTAL_SAMPLES,
    indices=VAL_INDICES
)

print("Validation dataset created.")


# ============================================================
# DataLoaders
# ============================================================

print("\nCreating DataLoaders...")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("DataLoaders created.")


# ============================================================
# Model
# ============================================================

print("\nCreating U-Net...")

model = UNet()

model = model.to(DEVICE)

print("U-Net created successfully.")


# ============================================================
# Loss function
# ============================================================

print("\nCreating loss function...")

criterion = CombinedLoss(
    dice_weight=0.5,
    flood_weight=2.0
)

print("Loss function created.")


# ============================================================
# Optimizer
# ============================================================

print("\nCreating optimizer...")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

print("Optimizer created.")


# ============================================================
# Training
# ============================================================

print("\nStarting training...")
print("=" * 60)


for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 60)

    model.train()

    total_train_loss = 0.0


    # ========================================================
    # Training batches
    # ========================================================

    for batch_index, batch in enumerate(train_loader):

        print(
            f"\nTraining batch "
            f"{batch_index + 1}/{len(train_loader)}"
        )

        images, masks, valid_masks = batch

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        valid_masks = valid_masks.to(DEVICE)


        # ----------------------------------------
        # Forward pass
        # ----------------------------------------

        outputs = model(images)


        # ----------------------------------------
        # Loss
        # ----------------------------------------

        loss = criterion(
            outputs,
            masks,
            valid_masks
        )


        # ----------------------------------------
        # Backpropagation
        # ----------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ----------------------------------------
        # Track loss
        # ----------------------------------------

        total_train_loss += loss.item()

        print(
            f"Loss: {loss.item():.4f}"
        )


    # ========================================================
    # Average training loss
    # ========================================================

    average_train_loss = (
        total_train_loss
        / len(train_loader)
    )

    print(
        f"\nAverage Training Loss: "
        f"{average_train_loss:.4f}"
    )


    # ========================================================
    # Validation
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Validation")
    print("=" * 60)

    model.eval()

    total_val_loss = 0.0

    total_dice = 0.0

    total_iou = 0.0


    with torch.no_grad():

        for batch_index, batch in enumerate(val_loader):

            images, masks, valid_masks = batch

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            valid_masks = valid_masks.to(DEVICE)


            # ------------------------------------
            # Prediction
            # ------------------------------------

            outputs = model(images)


            # ------------------------------------
            # Validation loss
            # ------------------------------------

            loss = criterion(
                outputs,
                masks,
                valid_masks
            )

            total_val_loss += loss.item()


            # ------------------------------------
            # Dice
            # ------------------------------------

            dice = calculate_dice(
                outputs,
                masks,
                valid_masks
            )


            # ------------------------------------
            # IoU
            # ------------------------------------

            iou = calculate_iou(
                outputs,
                masks,
                valid_masks
            )


            total_dice += dice

            total_iou += iou


            print(
                f"Validation batch "
                f"{batch_index + 1}/{len(val_loader)} "
                f"| Loss: {loss.item():.4f} "
                f"| Dice: {dice:.4f} "
                f"| IoU: {iou:.4f}"
            )


    # ========================================================
    # Average validation metrics
    # ========================================================

    average_val_loss = (
        total_val_loss
        / len(val_loader)
    )

    average_dice = (
        total_dice
        / len(val_loader)
    )

    average_iou = (
        total_iou
        / len(val_loader)
    )


    # ========================================================
    # Validation Results
    # ========================================================

    print("\n")
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)

    print(
        f"Average Validation Loss : "
        f"{average_val_loss:.4f}"
    )

    print(
        f"Average Dice Score      : "
        f"{average_dice:.4f}"
    )

    print(
        f"Average IoU Score       : "
        f"{average_iou:.4f}"
    )


    # ========================================================
    # Save best model
    # ========================================================

    if average_dice > best_dice:

        best_dice = average_dice

        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            "models/geopulse_unet_best.pth"
        )

        print("\n🏆 New best model!")

        print(
            f"Best Dice: "
            f"{best_dice:.4f}"
        )

        print(
            f"Best Epoch: "
            f"{best_epoch}"
        )


# ============================================================
# Training complete
# ============================================================

print("\n")
print("=" * 60)
print("Training complete!")
print("=" * 60)

print(
    f"Best Epoch : "
    f"{best_epoch}"
)

print(
    f"Best Dice  : "
    f"{best_dice:.4f}"
)

print(
    "Best model : "
    "models/geopulse_unet_best.pth"
)