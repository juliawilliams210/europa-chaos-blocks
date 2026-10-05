import os
import cv2
import json
import numpy as np
import rasterio


def build_coco_json(
    img_dir: str,
    mask_dir: str,
    out_json_dir: str
):
    """
    Reads pairs of image and mask tiles and generates a COCO formatted JSON.
    Assumes image and mask files have exactly the same name (e.g., tile_01.tif)

    Parameters:
    -----------
    img_dir : str
        Path to directory containing image tiles.

    mask_dir : str
        Path to directory containing mask tiles.

    out_json_dir : str
        Directory for storing COCO formatted JSON files.
    """
    # define the COCO JSON format
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "chaos_block", "supercategory": "geology"}
        ]
    }

    # initialize image and annotation ID values
    annotation_id = 1
    image_id = 1

    # loop through all masks in your tiled folder
    for filename in sorted(os.listdir(mask_dir)):
        if not filename.endswith((".tif", ".png", ".jpg")):
            continue

        mask_path = os.path.join(mask_dir, filename)

        # read in the mask
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            height, width = mask.shape

        # add the image info to COCO
        coco_format["images"].append({
            "id": image_id,
            "file_name": filename,  # just the file name
            "width": width,
            "height": height
        })

        # find unique blocks
        block_ids = np.unique(mask)
        block_ids = block_ids[block_ids != 0]  # ignore background

        for b_id in block_ids:
            single_block_mask = np.uint8(mask == b_id)

            # find contours
            contours, _ = cv2.findContours(
                single_block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10:  # skip tiny slivers
                    continue

                # get the bounding box [x_min, y_min, width, height]
                x, y, w, h = cv2.boundingRect(contour)

                # turn contour coordinates into 1D list: [x1, y1, x2, y2, ...]
                segmentation = contour.flatten().tolist()

                # a valid COCO polygon needs at least 6 coordinates
                if len(segmentation) >= 6:
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "segmentation": [segmentation],
                        "bbox": [x, y, w, h],
                        "area": area,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        image_id += 1

    # save the dictionary as a JSON file
    with open(out_json_dir, "w") as f:
        json.dump(coco_format, f, indent=4)

    print(f"Saved COCO JSON to {out_json_dir} with {image_id-1} images")
    print(f"and {annotation_id-1} annotations.")
