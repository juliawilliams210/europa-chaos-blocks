import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.features import rasterize
from shapely.geometry import box


def build_image_index(img_dir):
    """
    Build as index of all the tiff image files in a directory with
    their paths, bounds, and CRS.

    Parameters:
    -----------
    img_dir : str
        Path to the directory containing the TIFF images.

    Returns:
    --------
    img_records : list of dict
        A list of dictionaries containing the paths, bounds, and
        CRS of each TIFF image.
    """
    img_records = []
    # store the paths, bounds, and CRS of every TIFF in a list
    for tif in sorted(glob.glob(os.path.join(img_dir, "*.tif"))):
        with rasterio.open(tif) as src:
            img_records.append({
                "tiff_path": tif,
                "crs": src.crs,
                "bounds": box(*src.bounds)
            })
    return img_records


def stitch_images_and_masks(
        geojson_dir: str,
        img_dir: str,
        out_dir: str,
        region_crs: str = "ESRI:104915",
        padding_factor: float = 0.5,
        plates_only: bool = False,
):
    """
    Stitches intersecting TIFFs around a GeoJSON region and generates a uint16
    Instance Mask.

    Parameters:
    -----------
    geojson_dir : str
        Path to the directory containing the geojson files of block outlines.

    img_dir : str
        Path to the directory containing TIFF images of Europa

    out_dir : str
        Path to the output directory for storing the stitched images and masks.

    region_crs : str, optional
        The CRS to use for the geojson regions. The default is "ESRI:104915".

    padding_factor: float, optional
        The factor by which to pad the stitched images around the chaos
        region bounds. The default is 0.5, meaning the stitched image will be
        padded by 50% of the regions width and height.

    plates_only: bool, optional
        If True, only include objects labelled as plates. Default is False.

    Returns:
    --------
    None
    """
    # create the output directories if they dont exist
    img_out_dir = os.path.join(out_dir, "images")
    mask_out_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    tiffs = build_image_index(img_dir)

    # loop through the geojson files
    for json in sorted(glob.glob(os.path.join(geojson_dir, "*.json"))):

        # read in the json file and get the region name
        region = gpd.read_file(json)
        region_name = os.path.splitext(os.path.basename(json))[0]

        if plates_only is True:
            if 'Block_Plat' in region.columns:
                region = region[region['Block_Plat'] == 'Plate']

                if region.empty:
                    print(f"[SKIP] {region_name} has no Plates.")
                    continue

            if region.crs is None:
                region = region.set_crs(region_crs)

        # find all TIFFs that intersect this region
        intersecting_tiffs = []
        target_crs = None

        for tinfo in tiffs:
            tif = tinfo["tiff_path"]
            tif_crs = tinfo["crs"]

            # project region to TIFF CRS to check intersection
            region_proj = region.to_crs(tif_crs) if tif_crs else region
            if box(*region_proj.total_bounds).intersects(tinfo["bounds"]):
                intersecting_tiffs.append(tif)
                target_crs = tif_crs

        if not intersecting_tiffs:
            print(f"No intersecting TIFFs found for {region_name}")
            continue

        # ensure region is in the final target CRS
        region = region.to_crs(target_crs) if target_crs else region

        # add padding to the stitched image to capture background
        minx, miny, maxx, maxy = region.total_bounds
        pad_x, pad_y = (
            (maxx - minx) * padding_factor, (maxy - miny) * padding_factor
        )
        stitch_bounds = (
            minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y
        )

        # open the TIFFs and stitch them together
        srcs_to_close = []
        try:
            for tif in intersecting_tiffs:
                srcs_to_close.append(rasterio.open(tif))

            # merge stitches the arrays and calculates the new transform
            mosaic, out_trans = merge(
                srcs_to_close, bounds=stitch_bounds, nodata=np.nan
            )
        finally:
            # close the files to free up memory
            for src in srcs_to_close:
                src.close()

        # clean the stitched image data
        img_data = mosaic[0].astype(np.float32)
        img_data[img_data < -1e30] = np.nan
        img_data[img_data > 1e30] = np.nan

        if np.isnan(img_data).all():
            print(f"[WARN] Stitched area is empty NoData for {region_name}")
            continue

        # rasterize the json into a binary mask
        shapes = (
            (geom, index + 1) for index, geom in enumerate(region.geometry)
        )
        mask = rasterize(
            shapes=shapes,
            out_shape=img_data.shape,
            transform=out_trans,
            fill=0,
            dtype='uint16'
        )

        # save the stitched image as a geoTIFF
        img_out_path = os.path.join(img_out_dir, f"{region_name}.tif")
        with rasterio.open(
            img_out_path,
            'w',
            driver='GTiff',
            height=img_data.shape[0],
            width=img_data.shape[1],
            count=1,
            dtype=img_data.dtype,
            crs=target_crs,
            transform=out_trans,
            nodata=np.nan
        ) as dest:
            dest.write(img_data, 1)

        # save the mask as a geoTIFF
        mask_out_path = os.path.join(mask_out_dir, f"{region_name}.tif")
        with rasterio.open(
            mask_out_path,
            'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype=mask.dtype,
            crs=target_crs,
            transform=out_trans
        ) as dest:
            dest.write(mask, 1)

        print(f"saved {region_name} -> {img_out_path} & {mask_out_path}")
