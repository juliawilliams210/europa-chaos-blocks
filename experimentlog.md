# Model Training Log

| Run ID | Date | Model | Epochs | Best Epoch | Augmentations | Best mAP50 | Overfit Point | Notes | Next Step |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `run_01_yolo8n` | 10-06-2026 | `yolov8n-seg` | 50 | ~40 | Flip 'n' Slide | ~0.10 | ~42 | Dropped sharply at end. | Upgrade to larger model. |
| `run_02_yolo8s` | 10-06-2026 | `yolov8s-seg` | 100 | ~30 | Flip 'n' Slide & Mosaic ON | 0.08 | ~30 | Severe overfitting at epoch 30. | Apply heavy augmentations. |
| `run_03_yolo26s` | 10-07-2026 | `yolo26s-seg` | 50 |  | Flip 'n' Slide & Mosaic ON |  |  | Severe overfitting at epoch 30. | Apply heavy augmentations. |