import torch


def calculate_dice(
    logits,
    targets,
    valid_mask,
    threshold=0.5
):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.unsqueeze(1)

    valid_mask = valid_mask.unsqueeze(1)

    predictions = predictions * valid_mask

    targets = targets * valid_mask

    intersection = (
        predictions * targets
    ).sum()

    total = (
        predictions.sum()
        + targets.sum()
    )

    dice = (
        (2.0 * intersection + 1.0)
        /
        (total + 1.0)
    )

    return dice.item()


def calculate_iou(
    logits,
    targets,
    valid_mask,
    threshold=0.5
):

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    targets = targets.unsqueeze(1)

    valid_mask = valid_mask.unsqueeze(1)

    predictions = predictions * valid_mask

    targets = targets * valid_mask

    intersection = (
        predictions * targets
    ).sum()

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum()

    iou = (
        (intersection + 1.0)
        /
        (union + 1.0)
    )

    return iou.item()