# GeoPulse results

Model: U-Net (2 channels: VV, VH), trained 30 epochs on the 252-tile Sen1Floods11 hand-labelled
training split with flips/rotations (runs/aug/best.pth, seed 42, single run).

Test split (90 tiles), pooled over valid pixels, threshold 0.40 chosen on the validation set:

| Precision | Recall | Dice | IoU |
|---|---|---|---|
| 0.772 | 0.807 | 0.789 | 0.652 |

Notes: single training run (no error bars); pooled metrics, not per-tile; no independent
flood map exists for the Patna September 2026 scenes, so those areas are model estimates.

Bolivia split (a flood event held out of the dataset's train/valid/test splits), threshold 0.40 fixed in advance, not tuned on this split:

| Precision | Recall | Dice | IoU |
|---|---|---|---|
| 0.851 | 0.770 | 0.808 | 0.678 |

This is a single event with 15 tiles, so it does not show how the model behaves across regions. The best
threshold on this event (0.20) differs from the one on the test split (0.55), so the threshold should not
be assumed to transfer.
