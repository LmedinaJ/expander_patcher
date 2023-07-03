from scipy import ndimage
from pathlib import Path
from glob import glob
import numpy as np
import rasterio as rio


def getExpanderPatcher(image):
    """
    Identify the expand and patcher pixels in a image and
    returns an image with values between 1 and 2.

    Parameters:
    image : an image with pixel values between 1 and 2 representing the transitions.

    Returns:
    numpy.ndarray: an image with pixel values between 1 and 2 where pixels with
    value of 1 represent the expander and value of 2 the patcher.
    """

    binary_image = image == 1

    structure = ndimage.generate_binary_structure(2, 2)

    dilated_image = ndimage.binary_dilation(binary_image, structure=structure)

    adjacent_pixels = (image == 2) & dilated_image

    labeled = ndimage.label(image == 2, structure=structure)[0]

    adjacent_labels = set(labeled[adjacent_pixels])

    result = np.where(
        np.isin(labeled, list(adjacent_labels)), 1, np.where(image == 2, 2, 0)
    )

    result_uint8 = result.astype(np.uint8)

    return result_uint8


def main(rasters, path_out):
    """
    Process a list of rasters files in a folder.

    Parameters:
    rasters : list of raster files to process
    path_out: folder where the processed raster files are going to be saved

    Returns:
    void
    """

    for raster in rasters:
        print("procesando :", raster)

        with rio.open(raster) as src:
            image = src.read(1)
            crs = src.crs
            transform = src.transform

        result_uint8 = getExpanderPatcher(image)

        with rio.open(
            f"{path_out}expander_patcher_{raster.split('/')[-1]}",
            "w",
            driver="GTiff",
            height=result_uint8.shape[0],
            width=result_uint8.shape[1],
            count=1,
            dtype=result_uint8.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(result_uint8, 1)


if __name__ == "__main__":
    path = "raster_process/"

    path_out = "output_raster/"

    Path(path_out).mkdir(parents=True, exist_ok=True)

    rasters = glob(path + "*.tif")

    main(rasters, path_out)
