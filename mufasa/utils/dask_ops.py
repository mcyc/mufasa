"""
This module provides tools and wrappers to perform dask operations
that can't be performed in short-handed way like numpy arrays

"""
import dask.array as da
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import median_abs_deviation

from .. import aic
from . import dask_utils


def _masked_block_operate_old(a_block, b_block, mask_block, func):
    """
    Perform an operation on data and model blocks where the mask is True, using a custom function.

    Parameters:
        a_block (ndarray): A chunk of the data array.
        b_block (ndarray): A chunk of the model array.
        mask_block (ndarray): A chunk of the boolean mask array.
        func (callable): A function that takes two ndarrays (data and model) and returns an ndarray.

    Returns:
        ndarray: Resulting chunk after applying the function where the mask is True.
    """
    if not mask_block.any():
        # Skip computation if mask is entirely False in this block
        return np.zeros_like(a_block)

    # Apply the function only where the mask is True
    result = func(a_block, b_block)
    return np.where(mask_block, result, 0)


def _block_operate(block_mask, func, *blocks):
    if not block_mask.any():
        # Skip computation if the mask is entirely False in this block
        return np.zeros_like(block_mask, dtype=blocks[0].dtype)
    return func(*blocks)


def _masked_block_operate(block_mask, func, *blocks):
    """
    Helper function to apply the custom function `func` on blocks of data
    where the mask is True.

    Parameters:
        block_mask (np.ndarray): The mask for the current block.
        func (callable): The custom function to apply.
        *blocks (np.ndarray): The data blocks to which the function is applied.

    Returns:
        np.ndarray: The result of applying the function element-wise where the mask is True.
    """
    if not block_mask.any():
        # Skip computation if the mask is entirely False in this block
        return np.zeros_like(block_mask, dtype=blocks[0].dtype)

    # Apply the function to the blocks where the mask is True
    result = np.where(block_mask, func(*blocks), 0)
    return result


def masked_operate(darrays, mask, func, fullbock=False):
    """
    Apply a custom function element-wise on a Dask array or a tuple of Dask arrays (darrays)
    where mask is True.

    Parameters:
        darrays (dask.array.Array or tuple of dask.array.Array): Dask array(s) to apply the operation to.
        mask (dask.array.Array): A boolean Dask array acting as a mask.
        func (callable): A function that takes numpy array(s) as input and returns a numpy array.

    Returns:
        dask.array.Array: The result of applying the operation.
    """
    # Ensure darrays is a tuple if it's a single Dask array
    if not isinstance(darrays, tuple):
        darrays = (darrays,)

    # Ensure mask is a Dask array
    if not isinstance(mask, da.Array):
        raise ValueError("Mask must be a Dask array")

    # Broadcast the mask to match the shape of the first Dask array
    if mask.shape != darrays[0].shape:
        mask = da.broadcast_to(mask, darrays[0].shape)

    # Ensure mask has the same chunking as the first Dask array
    if mask.chunks != darrays[0].chunks:
        mask = mask.rechunk(darrays[0].chunks)

    # Apply map_blocks
    if fullbock:
        # no masking within the block
        result = da.map_blocks(
            _block_operate,
            mask,  # The first positional argument (block_mask)
            func,  # The second positional argument (func)
            *darrays,  # Additional blocks
            dtype=darrays[0].dtype,
        )
    else:
        # ensure mask is applied within the block
        result = da.map_blocks(
            _masked_block_operate,
            mask,  # The first positional argument (block_mask)
            func,  # The second positional argument (func)
            *darrays,  # Additional blocks
            dtype=darrays[0].dtype,
        )
    return result


def residual(cube, model, mask, fullblock=True):
    def sub(cube, model):
        return cube - model

    return masked_operate(darrays=(cube, model), mask=mask, func=sub, fullbock=fullblock)


def get_noise_AICc(cube, mask, return_rss=False):

    p = 0 #no free parameter for noise

    # get sample size
    nsamp = da.sum(mask, axis=0)
    def sq(res):
        return res**2
    rs = masked_operate(cube, mask, func=sq)
    rss = da.sum(rs * mask, axis=0)
    AICc = aic.AICc(rss, p, nsamp)
    AICc = AICc.compute()
    if return_rss:
        rss = rss.compute()
        return AICc, rss
    else:
        return AICc




def get_model_stats(cube, model, mask, p=None):
    # return AICc if p is provided

    # get sample size
    nsamp = da.sum(mask, axis=0)

    # get the residual (no mask)
    res = residual(cube, model, mask, fullblock=True)

    # get residual squared sum
    def sq(res):
        return res**2
    rs = masked_operate(res, mask, func=sq)
    rss = da.sum(rs * mask, axis=0)
    rss = da.where(rss == 0, np.nan, rss) # replace zero values

    # place nan
    nsamp = da.where(nsamp==0, np.nan, nsamp.astype(float))

    rss = rss.compute()
    nsamp = nsamp.compute()

    # approximate reduced chi-squared, without masking
    rms = residual_rms(res)
    rms = da.where(da.isnan(nsamp), np.nan, rms)

    # calculate reduced chisquared
    chisq_rd = rss / rms ** 2 / nsamp

    rms = rms.compute()
    chisq_rd = chisq_rd.compute()

    if p is None:
        # no AICc calculated
        return rss, nsamp, rms, chisq_rd, rms

    else:
        AICc = aic.AICc(rss, p, nsamp)
        return rss, nsamp, rms, chisq_rd, AICc


def residual_rms(residual, method='roll'):
    # Define the MAD function

    if method == 'roll':
        #Use effecient, yet robust method that assumes the median is zero
        def mad_std(residual, axis=None):
            diff = residual - np.roll(residual, 2, axis=axis)
            return 1.4826 * np.nanmedian(np.abs(diff), axis=axis) / 2 ** 0.5

    elif method == 'standard':
        # use scipy's MAD
        def mad_std(residual, axis=None):
            return median_abs_deviation(residual, axis=axis, scale='normal')

    # Apply scipy.stats.median_abs_deviation using map_blocks
    result = da.map_blocks(
        mad_std,
        residual,
        axis=0,  # Pass axis to the function
        dtype=residual.dtype,  # Ensure the correct dtype is specified
        drop_axis=0
    )
    return result


def padded_model_mask(model, pad=10, include_nosamp=True, planemask=None):
    mask = model > 0
    return pad_mask(mask, pad=pad, include_nosamp=include_nosamp, planemask=planemask)


def pad_mask(mask, pad=10, include_nosamp=True, planemask=None):

    if include_nosamp:
        # fill nosamp pixels with the sum of the existing spectral mask

        nosamp = ~da.any(mask, axis=0)
        def fill_mask(mask, nosamp):
            # operate per block, using the spectral mask sum of hte block
            specmask_fill = np.any(mask, axis=(1, 2))
            print(f"spec_fill sum: {np.sum(specmask_fill)}")
            mask[:, nosamp] = specmask_fill[:, np.newaxis]
            return mask

        if planemask is None:
            # assume the entire cube is good
            opmask = da.ones_like(mask, dtype=bool)

        else:
            # Expand the footprint mask to 3D
            if isinstance(planemask, np.ndarray):
                # Convert planemask to a Dask array with same chunking as the model
                planemask = da.from_array(planemask, chunks=(mask.chunks[1], mask.chunks[2]))

            # expand planemask along the spectral axis
            opmask = planemask[np.newaxis, :, :]

        # fill the mask over where no samp is with
        mask = masked_operate(darrays=(mask, nosamp), mask=opmask, func=fill_mask, fullbock=True)

    # could use more block effeciency in the future
    selem = np.ones(pad, dtype=bool)
    selem.shape += (1, 1,)
    mask = dask_binary_dilation(mask, selem)
    return dask_utils.persist_and_clean(mask)


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

    This operates similar to cube[:, planemask]

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
