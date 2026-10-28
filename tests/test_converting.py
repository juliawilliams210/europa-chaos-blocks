import os
import json
import tempfile
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.converting import build_coco_json


def create_dummy_mask(path, data):
    """Helper function to create a fake TIFF mask for testing."""
    transform = from_origin(0, 100, 1, 1)
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs='EPSG:3857', transform=transform
    ) as dest:
        dest.write(data, 1)


def test_build_coco_json():
    """
    Test that the COCO JSON builder correctly reads TIFF masks,
    filters out tiny slivers, creates the proper bounding boxes,
    and replaces the .tif extension with .jpg.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # set up temporary paths
        img_dir = os.path.join(tmpdir, "images")
        mask_dir = os.path.join(tmpdir, "masks")
        os.makedirs(img_dir)
        os.makedirs(mask_dir)

        out_json_path = os.path.join(tmpdir, "test_coco.json")

        # create a fake mask with two blocks
        mask_data = np.zeros((100, 100), dtype=np.uint8)

        mask_data[10:30, 10:30] = 1
        mask_data[50:52, 50:52] = 2

        mask_file = os.path.join(mask_dir, "tile_01.tif")
        create_dummy_mask(mask_file, mask_data)

        # run the function
        build_coco_json(
            img_dir=img_dir,
            mask_dir=mask_dir,
            out_json_dir=out_json_path
        )

        # check that the JSON was actually created
        assert os.path.exists(out_json_path), "JSON file was not created!"

        # B: Load the JSON to inspect its contents
        with open(out_json_path, 'r') as f:
            coco = json.load(f)

        # check general structure
        assert "images" in coco, "Missing 'images' key in JSON"
        assert "annotations" in coco, "Missing 'annotations' key in JSON"
        assert "categories" in coco, "Missing 'categories' key in JSON"

        # check image formatting
        assert len(coco["images"]) == 1, "Expected exactly 1 image entry"
        img_info = coco["images"][0]
        assert img_info["file_name"] == "tile_01.jpg", (
            "Did not replace .tif with .jpg correctly!"
        )
        assert img_info["width"] == 100, "Width should be 100"

        # check annotation filtering and math
        assert len(coco["annotations"]) == 1, (
            f"Expected 1 annotation, but found {len(coco['annotations'])}. "
            "The area filter (< 10) failed!"
        )

        ann = coco["annotations"][0]
        # the bounding box should be [x_min, y_min, width, height]
        assert ann["bbox"] == [10, 10, 20, 20], "Bounding box math is wrong!"
        assert ann["category_id"] == 1, "Category ID should be 1"
