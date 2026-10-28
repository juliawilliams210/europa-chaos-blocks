import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.sam_dataset import ChaosSAMDataset
from src.sam_builder import build_finetunable_sam


def run_sam_training(
    img_dir: str,
    mask_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    device: str = "mps"
):
    """
    Fine-tunes the SAM Mask Decoder on a custom dataset.
    """
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | LR: {learning_rate}")

    # load model and data
    model, processor = build_finetunable_sam(device=device)

    train_dataset = ChaosSAMDataset(img_dir, mask_dir, processor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    # optimizer and loss functions
    optimizer = optim.Adam(
        model.mask_decoder.parameters(), lr=learning_rate, weight_decay=0.0001
    )
    criterion = nn.BCEWithLogitsLoss()

    # traiing loop
    print(f"\nStarting SAM Fine-Tuning on {len(train_dataset)} images...")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        # disable tqdm in Jupyter notebooks to prevent messy, glitchy printouts
        # instead, we just print a clean status update per epoch
        batch_count = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(torch.float32).to(device)
            input_boxes = batch["input_boxes"].to(torch.float32).to(device)
            ground_truth = batch["ground_truth_mask"].to(
                torch.float32).to(device)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values,
                input_boxes=input_boxes,
                multimask_output=False
            )

            predicted_masks = outputs.pred_masks.squeeze(1).squeeze(1)

            loss = criterion(predicted_masks, ground_truth)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # print an update every 100 batches so you know it hasn't crashed
            if batch_count % 100 == 0:
                print(
                    f"Batch {batch_count}/{len(train_loader)} -"
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        print(
            f"Epoch [{epoch+1}/{epochs}] Complete."
            f"Average Loss: {avg_loss:.4f}\n"
        )

    # save
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "chaos_decoder.pth")
    torch.save(model.mask_decoder.state_dict(), weights_path)

    print(f"Training Complete. Fine-tuned weights saved to:\n{weights_path}")
