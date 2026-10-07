# Model Training Log

| Run ID | Date | Model | Epochs | Batch | Augmentations | Additional Params | Best Epoch | Best mAP50 (B) | Best mAP50 (M)| Overfit Point | Notes | Next Step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `run_01_yolo8n` | 10-06-2026 | `yolov8n-seg` | 50 | 16 | Flip 'n' Slide | None | ~40 | ~0.105 | ~0.08 | ~20 | Dropped sharply at end. | Upgrade to larger model. |
| `run_02_yolo8s` | 10-06-2026 | `yolov8s-seg` | 100 | 16 | Flip 'n' Slide, `close_mosaic=0` | None | ~30 | ~0.105 | ~0.08 | ~20 | Overfitting at epoch 30. | Try newer model version. |
| `run_03_yolo26s` | 10-07-2026 | `yolo26s-seg` | 50 | 16 | Flip 'n' Slide, `close_mosaic=0` | None | ~42 | ~0.11 | ~0.085 | ~30 | Loss curves flat & stable. Accuracy hit a hard ceiling. | Apply additional settings to deal with overlapping labels |
| `run_04_yolo26s` | 10-07-2026 | `yolo26s-seg` | 40 | 16 | Flip'n'Slide, `close_mosaic=0` |  `patience=15`, `overlap_mask=False`, `iou=0.7`, `box=7.5` | ~23 | ~0.12 | ~0.09 | N/A | Loss curves didn't really flatten out.| Try with only plates (larger blocks) |
| `run_05_yolo26s` | 10-07-2026 | `yolo26s-seg` | 40 | 16 | Flip'n'Slide, `close_mosaic=0` | PLATES ONLY,  `patience=15`, `overlap_mask=False`, `iou=0.7`, `box=7.5` | ~12 | ~0.09 | ~0.08 | N/A | Stopped early at 27 epochs due to no improvement. | |