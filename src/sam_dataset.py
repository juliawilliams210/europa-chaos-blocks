import os
import glob
import cv2
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset


class ChaosSAMDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.images = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = os.path.join(
            self.mask_dir, os.path.basename(img_path).replace('.jpg', '.tif')
        )

        # read image
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # read Mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        binary_mask = (mask > 0).astype(np.uint8)

        # generate a bounding box prompt for SAM to practice on
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # if there are no blocks, return a dummy box to prevent crashes
        if len(contours) == 0:
            prompt_box = [0, 0, img_rgb.shape[1], img_rgb.shape[0]]
            target_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]))
        else:
            # pick one random block from the image per iteration
            contour = contours[np.random.randint(0, len(contours))]
            x, y, w, h = cv2.boundingRect(contour)

            # add a tiny bit of random noise (jitter) to the box
            # this teaches SAM not to rely on perfect YOLO boxes
            x = max(0, x + np.random.randint(-2, 3))
            y = max(0, y + np.random.randint(-2, 3))
            prompt_box = [x, y, x + w, y + h]

            # create a mask specifically for this one block
            target_mask = cv2.drawContours(
                np.zeros_like(
                    binary_mask
                ), [contour], -1, 1, thickness=cv2.FILLED
                )

        # use hugging face processor to format everything for SAM
        inputs = self.processor(
            img_rgb,
            input_boxes=[[prompt_box]],
            return_tensors="pt"
        )

        # remove batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = torch.tensor(
            target_mask, dtype=torch.float32
        )

        return inputs
