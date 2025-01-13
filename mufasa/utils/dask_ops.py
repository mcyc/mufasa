"""
This module provides tools and wrappers to perform dask operations
that can't be performed in short-handed way like numpy arrays

"""
import dask.array as da
import numpy as np
from scipy.ndimage import binary_dilation

from . import dask_utils

def dask_binary_dilation(mask, selem):
    """
    Perform binary dilation on a Dask array using a specified structuring element.

    Parameters
    ----------
    mask : dask.array.Array
        The 2D Dask array to be dilated.
    selem : numpy.ndarray or dask.array.Array
        The structuring element used for dilation. Must be a 2D array.

    Returns
    -------
    dask.array.Array
        A Dask array representing the dilated mask.

    Raises
    ------
    ValueError
        If `selem` is not a NumPy or Dask array.

    Notes
    -----
    - The structuring element (`selem`) determines the neighborhood over which dilation
      is applied.
    - The function uses Dask's `map_overlap` to handle chunk-wise dilation with
      appropriate overlap for boundary regions.
    - The depth of overlap is automatically calculated as half the size of the
      structuring element along each dimension.

    Examples
    --------
    Apply binary dilation to a Dask array with a simple structuring element:

    >>> import dask.array as da
    >>> import numpy as np
    >>> from scipy.ndimage import generate_binary_structure

    >>> mask = da.from_array(np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]]), chunks=(3, 3))
    >>> selem = generate_binary_structure(2, 1)  # Cross-shaped structuring element

    >>> dilated_mask = dask_binary_dilation(mask, selem)
    >>> dilated_mask.compute()
    array([[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1]])
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

def apply_planemask(cube, planemask, persist=True):
    """
    Apply a 2D Boolean mask to a 3D cube, extracting pixel values within the mask.

    Parameters
    ----------
    cube : dask.array.Array
        A 3D Dask array of shape `(l, m, n)`, where `l` is the number of slices, and
        `(m, n)` are the spatial dimensions.
    planemask : numpy.ndarray or dask.array.Array
        A 2D Boolean mask of shape `(m, n)` specifying the spatial regions to include.
    persist : bool, default=True
        Whether to persist the resulting Dask array in memory for optimized access.

    Returns
    -------
    dask.array.Array
        A 2D Dask array of shape `(l, q)`, where `q = planemask.sum()` is the number
        of `True` values in the `planemask`.

    Raises
    ------
    ValueError
        If `cube` is not a 3D array, `planemask` is not a 2D array, or the spatial
        dimensions of `cube` and `planemask` do not match.

    Examples
    --------
    Apply a 2D planemask to a 3D cube:

    >>> import dask.array as da
    >>> import numpy as np
    >>> cube = da.random.random((3, 4, 5), chunks=(1, 2, 3))
    >>> planemask = np.array([[True, False, True, False, True],
    ...                       [False, True, False, True, False],
    ...                       [True, False, True, False, True],
    ...                       [False, True, False, True, False]])
    >>> result = apply_planemask(cube, planemask)
    >>> result.shape
    (3, 10)

    Handle invalid inputs:

    >>> invalid_planemask = np.array([[True, False]])
    >>> apply_planemask(cube, invalid_planemask)
    Traceback (most recent call last):
        ...
    ValueError: The spatial dimensions of `cube` and `planemask` must match.
    """
    # Validate inputs
    if cube.ndim != 3:
        raise ValueError("Input `cube` must be a 3D Dask array.")
    if planemask.ndim != 2:
        raise ValueError("Input `planemask` must be a 2D array.")
    if cube.shape[1:] != planemask.shape:
        raise ValueError("The spatial dimensions of `cube` and `planemask` must match.")

    # Flatten the spatial dimensions of the cube
    cube_flat = cube.reshape(cube.shape[0], -1)

    # Apply the planemask using da.compress
    cube_slice = da.compress(planemask.ravel(), cube_flat, axis=1)

    # Persist the result if requested
    return dask_utils.persist_and_clean(cube_slice) if persist else cube_slice
