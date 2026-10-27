import os
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from src.stitching import stitch_images_and_masks


# helper functions to build dummy data
def create_dummy_tiff(path):
    """Creates a fake 100x100 TIFF image filled with 1s."""
    data = np.ones((100, 100), dtype=np.float32)
    # origin at (0, 100), 1 meter per pixel
    transform = from_origin(0, 100, 1, 1)

    with rasterio.open(
        path, 'w', driver='GTiff',
        height=100, width=100, count=1,
        dtype=data.dtype, crs='EPSG:3857', transform=transform
    ) as dest:
        dest.write(data, 1)


def create_dummy_geojson(path):
    """Creates a fake block (20x20) in the middle of the image."""
    # a block from x(40-60), y(40-60)
    poly = Polygon([(40, 40), (60, 40), (60, 60), (40, 60)])
    # add 'Block_Plat' to test the plates_only=True
    gdf = gpd.GeoDataFrame(
        {'Block_Plat': ['Plate']}, geometry=[poly], crs='EPSG:3857'
    )
    gdf.to_file(path, driver='GeoJSON')


def create_dummy_shapefile(path, region_name):
    """Creates a fake boundary mask to test the cookie-cutter feature."""
    # draw a shapefile boundary from x(35 to 65).
    poly = Polygon([(35, 35), (65, 35), (65, 65), (35, 65)])

    gdf = gpd.GeoDataFrame(
        {'region': [region_name]}, geometry=[poly], crs='EPSG:3857'
    )
    gdf.to_file(path)


# pytest function
def test_stitch_and_mask_pipeline():
    """
    Tests that the stitching script successfully reads inputs,
    applies padding, burns masks, applies shapefile boundaries,
    and saves valid TIFF files.
    """
    # creates temporary directory that deletes itself when the test finishes
    with tempfile.TemporaryDirectory() as tmpdir:

        # setup paths
        img_dir = os.path.join(tmpdir, 'raw_images')
        json_dir = os.path.join(tmpdir, 'geojson')
        out_dir = os.path.join(tmpdir, 'output')
        shp_path = os.path.join(tmpdir, 'boundary.shp')

        os.makedirs(img_dir)
        os.makedirs(json_dir)

        # generate the dummy files
        create_dummy_tiff(os.path.join(img_dir, 'fake_europa.tif'))
        create_dummy_geojson(os.path.join(json_dir, 'Chaos_Test.json'))
        create_dummy_shapefile(shp_path, 'Chaos_Test')

        # run the function
        stitch_images_and_masks(
            geojson_dir=json_dir,
            img_dir=img_dir,
            out_dir=out_dir,
            boundary_shp_path=shp_path,
            region_crs="EPSG:3857",  # Use standard math for the test
            padding_factor=0.5,
            plates_only=True
        )

        out_img_path = os.path.join(out_dir, 'images', 'Chaos_Test.tif')
        out_mask_path = os.path.join(out_dir, 'masks', 'Chaos_Test.tif')

        # check that the files were actually created
        assert os.path.exists(out_img_path), "Output image was not created!"
        assert os.path.exists(out_mask_path), "Output mask was not created!"

        # open the output files to check their contents
        with (
            rasterio.open(out_img_path) as src_img,
            rasterio.open(out_mask_path) as src_mask
        ):

            img_data = src_img.read(1)
            mask_data = src_mask.read(1)

            # check that padding worked
            assert img_data.shape == (40, 40), (
                f"Expected padded shape (40, 40), got {img_data.shape}"
            )
            assert mask_data.shape == (40, 40), (
                "Mask shape does not match image shape!"
            )

            # check that the mask was burned properly
            assert np.max(mask_data) == 1, "The rasterized mask is empty!"

            # check that the shapefile cookie-cutter worked
            assert np.isnan(img_data).any(), (
                "The shapefile mask didn't apply any NaNs to the image!"
            )
