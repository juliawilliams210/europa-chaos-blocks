import os
import glob
import numpy as np
import rasterio
import tempfile

from src.tiling import generate_tiles


def create_dummy_geotiff(path, data):
    """Helper function to create fake TIFFs for testing"""
    with rasterio.open(
        path, 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype
    ) as dest:
        dest.write(data, 1)


def test_full_image_coverage():
    """
    Test that the tiling script completely covers a 500x500 image
    without leaving any missing gaps at the edges.
    """
    # create temporary directories for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "Chaos_Test.tif")
        mask_path = os.path.join(tmpdir, "Chaos_Test_mask.tif")
        out_img_dir = os.path.join(tmpdir, "images")
        out_mask_dir = os.path.join(tmpdir, "masks")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # create a fake 500x500 image filled it entirely with 1s
        dummy_img = np.ones((500, 500), dtype=np.float32)
        # create a mask filled it entirely with 1s
        dummy_mask = np.ones((500, 500), dtype=np.uint16)

        create_dummy_geotiff(img_path, dummy_img)
        create_dummy_geotiff(mask_path, dummy_mask)

        # run the tiling script
        generate_tiles(
            img_path=img_path,
            mask_path=mask_path,
            out_img_dir=out_img_dir,
            out_mask_dir=out_mask_dir,
            chip_size=256,
            stride=128,
            apply_flipnslide=False
        )

        # verify the coverage
        generated_masks = glob.glob(os.path.join(out_mask_dir, "*.tif"))

        # expect exactly 16 chips to cover a 500x500 area (4 rows x 4 cols)
        assert len(generated_masks) == 16, (
            f"Expected 16 chips, but got {len(generated_masks)}/16 chips"
        )
        # create a blank 500x500 canvas filled with 0s
        reconstructed_canvas = np.zeros((500, 500), dtype=np.uint16)
        canvas_h, canvas_w = reconstructed_canvas.shape

        # loop through all generated chips and stamp them back onto the canvas
        for mask_file in generated_masks:
            basename = os.path.basename(mask_file)

            # filename looks like 'Chaos_Test_128_256_orig.tif'
            parts = basename.split('_')
            y_start = int(parts[2])
            x_start = int(parts[3])

            with rasterio.open(mask_file) as src:
                chip_data = src.read(1)

            # calculate exactly how much space is left on the canvas
            y_end = min(y_start + 256, canvas_h)
            x_end = min(x_start + 256, canvas_w)

            # calculate how much of the chip we actually need to use
            chip_h = y_end - y_start
            chip_w = x_end - x_start

            # stamp only the valid portion of the chip onto the canvas
            reconstructed_canvas[y_start:y_end, x_start:x_end] = (
                chip_data[:chip_h, :chip_w]
            )

        # the entire canvas should now be filled with 1s
        missing_pixels = np.sum(reconstructed_canvas == 0)

        assert missing_pixels == 0, (
            f"{missing_pixels} pixels completely un-tiled."
        )
