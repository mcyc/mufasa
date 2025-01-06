import numpy as np
import dask.array as da
from dask import delayed
from dask.distributed import Client

#======================================================================================================================#
from .mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#

def calculate_chunks(cube_shape, dtype, target_chunk_mem_mb=128):
    """
    Calculate chunk sizes for a Dask array based on the given criteria.

    Parameters
    ----------
    cube_shape : tuple
        The shape of the data cube (spectral_axis, y, x).
    dtype : numpy.dtype
        The data type of the cube (e.g., np.float32, np.float64).
    target_chunk_mem_mb : int, optional
        Target memory size per chunk in MB. Default is 128 MB.

    Returns
    -------
    tuple
        Chunk sizes for the Dask array in the form (spectral_chunk, y_chunk, x_chunk).
    """
    spectral_axis, y_dim, x_dim = cube_shape
    dtype_size = np.dtype(dtype).itemsize  # Size of one element in bytes

    # Entire spectral axis in one chunk
    spectral_chunk = spectral_axis

    # Calculate memory per pixel along the spectral axis
    memory_per_pixel = spectral_axis * dtype_size

    # Total memory target for spatial chunk
    target_chunk_mem_bytes = target_chunk_mem_mb * 1024 * 1024

    # Number of pixels per chunk to fit in memory
    pixels_per_chunk = target_chunk_mem_bytes // memory_per_pixel

    # Determine spatial chunk size
    spatial_chunk_size = int(np.sqrt(pixels_per_chunk))  # Assume square chunks
    y_chunk = min(y_dim, spatial_chunk_size)
    x_chunk = min(x_dim, spatial_chunk_size)

    return (spectral_chunk, y_chunk, x_chunk)


def calculate_batch_size(spectral_size, dtype, total_valid_pixels, n_cores, memory_limit_mb=1024,
                         task_per_core=10, min_tpc=5):
    """
    Dynamically calculate the optimal batch size for processing valid pixels.

    This function computes the batch size based on memory constraints, the number of available workers,
    and the desired number of tasks per core. The goal is to balance memory usage and parallelism while
    ensuring efficient task distribution.

    Parameters
    ----------
    spectral_size : int
        Length of the spectral axis (z-dimension) of the data cube.
    dtype : numpy.dtype
        Data type of the cube (e.g., np.float32, np.float64). Used to calculate memory usage per pixel.
    total_valid_pixels : int
        Total number of valid pixels to process. These are the pixels specified by a mask or selection criteria.
    n_cores : int
        Number of workers available for parallel processing.
    memory_limit_mb : int, optional
        Approximate memory limit per worker in megabytes. Default is 1024 MB (1 GB).
    task_per_core : int, optional
        Target number of tasks per core to ensure sufficient parallelism. Default is 10 tasks per worker.
    min_tpc : int, optional
        Minimum number of tasks per core to maintain efficient distribution. Default is 5 tasks per worker.

    Returns
    -------
    tuple
        - int : Optimal batch size for processing valid pixels.
        - int : Adjusted number of workers to balance the workload and ensure efficient task distribution.

    Notes
    -----
    - The function calculates the maximum batch size that fits within the memory limit of a single worker.
    - It then adjusts the batch size and the number of workers based on the total number of valid pixels and the
      desired task-per-core ratio.
    - If the total number of tasks is insufficient to meet the minimum tasks per core, the function reduces the
      number of workers or sets the batch size to 0, signaling that the workload needs adjustment.

    Example
    -------
    >>> spectral_size = 1000
    >>> dtype = np.float32
    >>> total_valid_pixels = 10000
    >>> n_workers = 4
    >>> calculate_batch_size(spectral_size, dtype, total_valid_pixels, n_cores)
    (250, 4)
    """
    # Memory per spectrum in bytes
    memory_per_spectrum = spectral_size * np.dtype(dtype).itemsize

    # Memory available for batches per worker
    memory_per_worker = memory_limit_mb * 1024 * 1024

    # Maximum batch size per worker
    max_batch_size = memory_per_worker // memory_per_spectrum

    # Distribute tasks across all workers
    total_batches = n_cores * task_per_core  # Aim for 10 tasks per worker

    if total_batches < total_valid_pixels:
        # if there are enough batch jobs per core
        suggested_batch_size = total_valid_pixels // total_batches
    elif n_cores > 1:
        # reduce the number of workers
        n_cores_t = total_valid_pixels // min_tpc
        if n_cores_t > 0 and n_cores_t < n_cores:
            n_cores = n_cores_t
            suggested_batch_size = min_tpc
        else:
            suggested_batch_size = 0
    else:
        suggested_batch_size = 0

    # Use the smaller of memory-constrained or task-distributed batch size
    return min(max_batch_size, suggested_batch_size), n_cores


def lazy_pix_compute(host_cube, isvalid, compute_pixel):
    """
    Lazily compute values for each valid pixel specified by the isvalid mask.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array representing the data cube (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D boolean mask with the same spatial dimensions as `host_cube` (y, x).
        True indicates pixels to compute.
    compute_pixel : callable
        A function that computes values for a pixel. Should accept `(x, y)`
        as arguments and return a 1D array representing the spectral axis.

    Returns
    -------
    dask.array.Array
        A new Dask array with the same shape as `host_cube`, where the
        computed values for valid pixels are filled, and other values remain NaN.
    """
    # Ensure isvalid is a NumPy array for indexing
    if isinstance(isvalid, da.Array):
        isvalid = isvalid.compute()

    # Get the valid pixel indices from the mask
    valid_pixels = np.argwhere(isvalid)

    # Define a delayed computation for each valid pixel
    def compute_pixel_delayed(x, y):
        """Compute a single pixel lazily."""
        result = delayed(compute_pixel)(x, y)
        return da.from_delayed(result, shape=(host_cube.shape[0],), dtype=host_cube.dtype)

    # Initialize an empty Dask array with NaN
    computed_cube = da.full(host_cube.shape, np.nan, dtype=host_cube.dtype, chunks=host_cube.chunks)

    # Iterate through valid pixels and lazily update the computed_cube
    for y, x in valid_pixels:
        # Compute the value for the current pixel lazily
        pixel_value = compute_pixel_delayed(x, y)

        # Assign the computed value to the corresponding location in the cube
        computed_cube = da.map_blocks(
            lambda block, val=pixel_value, cx=x, cy=y: _update_pixel(block, val, cx, cy),
            computed_cube, dtype=host_cube.dtype
        )

    return computed_cube


def lazy_pix_compute_multiprocessing(host_cube, isvalid, compute_pixel, batch_size=100, n_workers=4):
    """
    Lazily compute values for valid pixels specified by the `isvalid` mask using multi-processing.

    This function processes pixels in batches for efficiency, with tasks distributed
    across multiple workers managed by a Dask client. The results are lazily computed
    and integrated into a Dask array, preserving memory efficiency.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array representing the data cube with dimensions (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D boolean mask of shape (y, x). Pixels marked as True will be processed,
        and their computed values will be updated in the output cube.
    compute_pixel : callable
        A function that computes the spectral data for a given pixel. It should accept
        `(x, y)` as arguments and return a 1D array of spectral values.
    batch_size : int, optional
        The number of pixels to process in each batch. Larger batches reduce task
        scheduling overhead but require more memory. Default is 100.
    n_workers : int, optional
        The number of workers to use for parallel processing. Default is 4.

    Returns
    -------
    dask.array.Array
        A Dask array with the same shape as `host_cube`, where computed values for valid
        pixels are filled, and all other locations remain NaN.

    Notes
    -----
    - The computation is performed in batches to balance parallelism and memory usage.
    - Tasks are distributed across workers using Dask's distributed scheduler.
    - Ensure the system has sufficient resources (CPU, memory) for the requested `n_workers`
      and `batch_size`.

    Examples
    --------
    >>> import numpy as np
    >>> import dask.array as da
    >>> from mymodule import lazy_pix_compute_multiprocessing
    >>>
    >>> # Example 3D Dask array and mask
    >>> cube = da.random.random((100, 50, 50), chunks=(50, 25, 25))
    >>> isvalid = np.zeros((50, 50), dtype=bool)
    >>> isvalid[10, 20] = True
    >>> isvalid[30, 40] = True
    >>>
    >>> # Example pixel computation function
    >>> def compute_pixel(x, y):
    >>>     return np.arange(100) + x + y  # Example: spectral values
    >>>
    >>> # Perform lazy computation
    >>> result = lazy_pix_compute_multiprocessing(cube, isvalid, compute_pixel, batch_size=50, n_workers=2)
    >>> print(result.compute().shape)
    (100, 50, 50)
    """

    # Ensure isvalid is a NumPy array for indexing
    if isinstance(isvalid, da.Array):
        isvalid = isvalid.compute()

    # Extract valid pixel indices
    valid_pixels = np.argwhere(isvalid)

    # Split valid pixels into batches
    pixel_batches = [
        valid_pixels[i : i + batch_size]
        for i in range(0, len(valid_pixels), batch_size)
    ]

    # Batch computation
    def compute_batch(batch):
        batch_data = np.full((host_cube.shape[0], len(batch)), np.nan, dtype=host_cube.dtype)
        for i, (y, x) in enumerate(batch):
            batch_data[:, i] = compute_pixel(x, y)
        return batch, batch_data

    # Initialize computed cube
    computed_cube = da.full(host_cube.shape, np.nan, dtype=host_cube.dtype, chunks=host_cube.chunks)

    try:
        # Initialize Dask client
        client = Client(n_workers=n_workers,
                        memory_limit="2GB",
                        local_directory="/tmp/dask-worker-space",
                        timeout="60s",
                        heartbeat_interval="30s")
        logger.info("Dask client initialized.")

        # Batch processing
        for batch in pixel_batches:
            delayed_result = delayed(compute_batch)(batch)
            computed_cube = da.map_blocks(
                lambda block, result=delayed_result: _update_batch(block, result.compute()),
                computed_cube,
                dtype=host_cube.dtype,
            )
    finally:
        client.close()

    return computed_cube

def _update_pixel(block, value, x, y):
    """
    Helper function to update a single pixel in a block.

    Parameters
    ----------
    block : np.ndarray
        The block of the Dask array being processed.
    value : np.ndarray
        The computed value for the pixel (1D array along the spectral axis).
    x : int
        The x-coordinate of the pixel within the block.
    y : int
        The y-coordinate of the pixel within the block.

    Returns
    -------
    np.ndarray
        A copy of the block with the pixel updated.
    """
    block_copy = block.copy()
    block_copy[:, y, x] = value
    return block_copy


def _update_batch(block, batch_result):
    """
    Update multiple pixels in a block based on batch results.

    Parameters
    ----------
    block : numpy.ndarray
        The block of the Dask array being processed.
    batch_result : tuple
        A tuple where:
        - batch_result[0] is a list of (y, x) pixel coordinates for the batch.
        - batch_result[1] is a 2D numpy.ndarray containing the computed spectral values
          for the batch. Shape: (spectral_axis, num_pixels).

    Returns
    -------
    numpy.ndarray
        Updated block with the batch results applied.
    """
    block_copy = block.copy()  # Create a writable copy
    batch_pixels, batch_data = batch_result

    for i, (y, x) in enumerate(batch_pixels):
        block_copy[:, y, x] = batch_data[:, i]  # Assign spectral values to each pixel

    return block_copy
