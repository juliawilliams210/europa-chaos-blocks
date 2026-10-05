import os
import numpy as np
import rasterio
from rasterio.windows import Window
import cv2


def generate_tiles(img_path: str,
                   mask_path: str,
                   out_img_dir: str,
                   out_mask_dir: str,
                   chip_size: int = 256,
                   stride: int = 128,
                   apply_flipnslide: bool = False):
    """
    Generate image and mask tiles from a given TIFF image and its
    corresponding TIFF mask. The tiles are saved to the specified output
    directories. Data augmentation using the "Flip'n'Slide" method can be
    optionally applied.

    Parameters:
    -----------
    img_path : str
        Path to the input TIFF image

    mask_path : str
        Path to the corresponding TIFF mask

    out_img_dir : str
        Output directory for storing generated image tiles

    out_mask_dir : str
        Output directory for storing generated mask tiles

    chip_size : int, optional
        Size of the square tiles to be generated (default is 256x256)

    stride : int, optional
        Stride for the sliding window (default is 128)

    apply_flipnslide : bool, optional
        Option to apply the "Flip'n'Slide" data augmentation method
        (default is False)
    """
    # determine the region name from the image path
    region_name = os.path.splitext(os.path.basename(img_path))[0]

    # use rasterio to read in the image and corresponding mask
    with (
        rasterio.open(img_path) as src_img,
        rasterio.open(mask_path) as src_mask
    ):

        # read the entire stitched images into memory
        full_img = src_img.read(1)
        full_mask = src_mask.read(1)

        # pad the image if it is smaller than tile size
        h, w = full_img.shape
        pad_h = max(0, chip_size - h)
        pad_w = max(0, chip_size - w)

        if pad_h > 0 or pad_w > 0:
            # pad the right and bottom edges with NaNs (image) and 0s (mask)
            full_img = np.pad(
                full_img, ((0, pad_h), (0, pad_w)),
                mode='constant', constant_values=np.nan
            )
            full_mask = np.pad(
                full_mask, ((0, pad_h), (0, pad_w)),
                mode='constant', constant_values=0
            )

        # update dimensions after potential padding
        height, width = full_img.shape

        # mask out the NaN values so they don't skew the math
        valid_pixels = full_img[~np.isnan(full_img)]
        if len(valid_pixels) == 0:
            return  # region is totally empty

        # calculate the percentiles for the whole image
        global_vmin, global_vmax = np.percentile(valid_pixels, [2, 98])

        chip_count = 0

        # start from the top-left corner and slide window across the image
        for y in range(0, height, stride):
            for x in range(0, width, stride):

                # do not generate a tile if it goes past the image bounds
                if y + chip_size > height or x + chip_size > width:
                    continue

                window = Window(x, y, chip_size, chip_size)
                img_chip = src_img.read(1, window=window)
                mask_chip = src_mask.read(1, window=window)

                # filter out tiles that are mostly NaN
                if np.isnan(img_chip).sum() > (chip_size * chip_size * 0.5):
                    continue

                # keep about 10% of background tiles to prevent class imbalance
                if np.max(mask_chip) == 0:
                    if np.random.rand() > 0.10:
                        continue

                # apply the global normalization to the image chip
                img_chip = np.nan_to_num(img_chip, nan=global_vmin)

                if global_vmax > global_vmin:
                    # clip prevents shadows or bright spots from being > 0 or 1
                    img_chip = np.clip(
                        (img_chip - global_vmin) / (global_vmax - global_vmin),
                        0, 1
                    )

                img_chip = (img_chip * 255).astype(np.uint8)

                # keep the original chip
                variations = {"orig": (img_chip, mask_chip)}

                # apply "Flip'n'Slide" if param is True
                if apply_flipnslide:

                    # apply rotations and flips to the chip
                    variations.update({
                        "rot90": (np.rot90(img_chip, 1), np.rot90(
                            mask_chip, 1)
                        ),
                        "rot180": (np.rot90(img_chip, 2), np.rot90(
                            mask_chip, 2)
                        ),
                        "rot270": (np.rot90(img_chip, 3), np.rot90(
                            mask_chip, 3)
                        ),
                        "flipLR": (np.fliplr(img_chip), np.fliplr(mask_chip)),
                        "flipUD": (np.flipud(img_chip), np.flipud(mask_chip))
                    })

                # save the variations
                for aug_name, (aug_img, aug_mask) in variations.items():
                    chip_name = f"{region_name}_{y}_{x}_{aug_name}"

                    cv2.imwrite(
                        os.path.join(out_img_dir, f"{chip_name}.jpg"), aug_img
                    )

                    out_mask_path = os.path.join(
                        out_mask_dir, f"{chip_name}.tif"
                    )
                    with rasterio.open(
                        out_mask_path, 'w', driver='GTiff',
                        height=chip_size, width=chip_size, count=1,
                        dtype=aug_mask.dtype
                    ) as dest:
                        dest.write(aug_mask, 1)

                    chip_count += 1

        # print the mode and number of chips generated for this region
        mode = "Flip'n'Slide" if apply_flipnslide else "Standard"

        print(f"Generated {chip_count} {mode} chips for {region_name}")
