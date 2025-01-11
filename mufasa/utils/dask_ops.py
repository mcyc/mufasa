import dask.array as da
import numpy as np
from scipy.ndimage import binary_dilation

from . import dask_utils

def dask_binary_dilation(mask, selem):
    """
    Perform binary dilation on a Dask array mask with a structuring element.

    Parameters:
    - mask: Dask array to be dilated
    - selem: Structuring element (can be a Dask array or NumPy array)

    Returns:
    - Dask array after binary dilation
    """
    # Ensure selem is a NumPy array, as binary_dilation requires it
    if isinstance(selem, da.Array):
        selem = selem.compute()
    elif not isinstance(selem, np.ndarray):
        raise ValueError("selem must be a Dask array or NumPy array.")

    # Define the dilation function
    def binary_dilation_func(chunk):
        return binary_dilation(chunk, structure=selem)

    # Apply binary dilation with map_overlap
    mask_dilated = mask.map_overlap(binary_dilation_func, depth=tuple(s // 2 for s in selem.shape), boundary="none")

    return dask_utils.persist_and_clean(mask_dilated, debug=False, visualize_filename=None)
