# GeoPulse results

Model: U-Net (2 channels: VV, VH), trained 30 epochs on the 252-tile Sen1Floods11 hand-labelled
training split with flips/rotations (runs/aug/best.pth, seed 42, single run).

Test split (90 tiles), pooled over valid pixels, threshold 0.40 chosen on the validation set:

| Precision | Recall | Dice | IoU |
|---|---|---|---|
| 0.772 | 0.807 | 0.789 | 0.652 |

Notes: single training run (no error bars); pooled metrics, not per-tile; no independent
flood map exists for the Patna September 2026 scenes, so those areas are model estimates.
