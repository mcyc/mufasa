import numpy as np
import dask.array as da
from dask import delayed, config
from dask.distributed import Client
import psutil

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


def lazy_pix_compute_single(host_cube, isvalid, compute_pixel):
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


def lazy_pix_compute_dynamic(host_cube, isvalid, compute_pixel, memory_limit_mb=1024, scheduler='adaptive'):
    """
    Adaptive computation of valid pixels with dynamic batching and scheduling.

    Parameters
    ----------
    host_cube : dask.array.Array
        The 3D data cube (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D boolean mask (y, x) where True indicates valid pixels.
    compute_pixel : callable
        A function that computes pixel spectral values given (x, y).
    memory_limit_mb : int, optional
        Memory limit per worker in MB. Default is 1024 MB.
    scheduler : {'adaptive', 'threads', 'processes', 'synchronous'}, default='adaptive'
        Scheduler to use for computation.

    Returns
    -------
    dask.array.Array
        A 3D array with computed values for valid pixels.
    """
    import dask.array as da
    from dask import config

    if isinstance(isvalid, da.Array):
        isvalid = isvalid.compute()

    # Calculate density of valid pixels
    valid_pixel_count = np.sum(isvalid)
    if valid_pixel_count == 0:
        raise ValueError("No valid pixels found in the mask. Ensure `isvalid` is correctly defined.")

    pixel_density = valid_pixel_count / isvalid.size

    # Determine number of physical cores
    n_cores = psutil.cpu_count(logical=False)
    logger.info(f"Detected physical cores: {n_cores}")

    # Determine optimal batch size dynamically
    spectral_size = host_cube.shape[0]
    dtype = host_cube.dtype
    batch_size, _ = calculate_batch_size(
        spectral_size,
        dtype,
        valid_pixel_count,
        n_cores=n_cores,  # Use physical cores
        memory_limit_mb=memory_limit_mb,
    )

    # Validate batch size
    if batch_size <= 0:
        logger.warning(
            f"Calculated batch size is invalid ({batch_size}). "
            f"Falling back to a default value based on valid pixels and cores."
        )
        batch_size = max(1, valid_pixel_count // n_cores)

    logger.info(f"Adaptive batch size: {batch_size}")

    # Dynamic scheduler
    if scheduler == 'adaptive':
        scheduler = 'threads' if pixel_density > 0.1 else 'processes'

    # Lazy computation of valid pixels
    computed_cube = da.full(
        host_cube.shape, np.nan, dtype=host_cube.dtype, chunks=host_cube.chunks
    )

    def compute_batch(batch_pixels):
        batch_data = np.full((host_cube.shape[0], len(batch_pixels)), np.nan, dtype=host_cube.dtype)
        for i, (y, x) in enumerate(batch_pixels):
            batch_data[:, i] = compute_pixel(x, y)
        return batch_pixels, batch_data

    # Process pixels in chunks or batches
    valid_pixels = np.argwhere(isvalid)
    pixel_batches = [
        valid_pixels[i : i + batch_size]
        for i in range(0, len(valid_pixels), batch_size)
    ]

    with config.set(scheduler=scheduler):
        for batch in pixel_batches:
            delayed_result = delayed(compute_batch)(batch)
            computed_cube = da.map_blocks(
                lambda block, result=delayed_result: _update_batch(block, result.compute()),
                computed_cube,
                dtype=host_cube.dtype,
            )

    return computed_cube


def lazy_pix_compute(host_cube, isvalid, compute_pixel, batch_size=100, n_workers=4, scheduler='threads'):
    """
    Lazily compute values for valid pixels specified by the `isvalid` mask using the specified scheduler.

    This function computes spectral values for a subset of pixels in a data cube,
    as determined by the `isvalid` mask. Pixels marked as valid (`True`) in the mask
    are processed using the specified `compute_pixel` function, while others are left as NaN.
    The computation is parallelized using Dask and can be scheduled using threads,
    processes, or batching with a Dask client.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array representing the data cube with dimensions (spectral, y, x).
        The cube contains the spectral data to be processed.

    isvalid : numpy.ndarray or dask.array.Array
        A 2D boolean mask of shape (y, x). Pixels marked as `True` will be processed,
        and their computed values will be updated in the output array. Pixels marked
        as `False` will remain as NaN.

    compute_pixel : callable
        A function that computes spectral data for a given pixel. It must accept two
        arguments, `(x, y)`, where `x` and `y` are the spatial coordinates of the pixel,
        and return a 1D array of spectral values.

    batch_size : int, optional, default=100
        The number of pixels to process in each batch. Larger batches reduce task
        scheduling overhead but require more memory.

    n_workers : int, optional, default=4
        The number of workers to use for parallel processing. This parameter is only
        used when the `scheduler` is set to `'batch'`. It determines the number of Dask
        workers launched in the local cluster.

    scheduler : {'threads', 'processes', 'synchronous', 'batch'}, optional, default='threads'
        The Dask scheduler to use for parallel computation. The available options are:
        - `'threads'`: Use a thread-based scheduler (default). Tasks run in parallel
          threads, sharing memory.
        - `'processes'`: Use a process-based scheduler. Tasks run in separate processes,
          each with its own memory space.
        - `'synchronous'`: Use a single-threaded, synchronous scheduler. Tasks are
          executed sequentially.
        - `'batch'`: Use batching with a Dask distributed client for more fine-grained
          control over the execution, including access to the Dask dashboard.

    Returns
    -------
    dask.array.Array
        A 3D Dask array with the same shape as `host_cube`. The computed values for
        valid pixels are filled into the array, while all other locations remain NaN.

    Notes
    -----
    - When the scheduler is set to `'batch'`, a local Dask distributed client is
      initialized to manage the computation. This provides access to the Dask dashboard
      for real-time task monitoring.
    - Ensure that `compute_pixel` is thread-safe if using the `'threads'` scheduler.

    Examples
    --------
    Compute spectral values for a subset of pixels in a data cube:

    >>> import dask.array as da
    >>> import numpy as np

    >>> def mock_compute_pixel(x, y):
    ...     return np.array([x + y, x - y])

    >>> host_cube = da.random.random((10, 100, 100), chunks=(10, 50, 50))
    >>> isvalid = np.random.choice([True, False], size=(100, 100))

    >>> result = lazy_pix_compute(
    ...     host_cube,
    ...     isvalid,
    ...     compute_pixel=mock_compute_pixel,
    ...     batch_size=10,
    ...     scheduler='threads'
    ... )
    >>> print(result.compute())  # Compute the result
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

    logger.info(f"Using scheduler: {scheduler}")
    if scheduler !='batch':
        # Use the non-batch scheduler
        with config.set(scheduler=scheduler):
            for batch in pixel_batches:
                delayed_result = delayed(compute_batch)(batch)
                computed_cube = da.map_blocks(
                    lambda block, result=delayed_result: _update_batch(block, result.compute()),
                    computed_cube,
                    dtype=host_cube.dtype,
                )

    else:
        # use batching
        try:
            # Initialize Dask client with a dashboard
            client = Client(
                n_workers=n_workers,
                memory_limit="2GB",
                local_directory="/tmp/dask-worker-space",
                timeout="60s",
                heartbeat_interval="30s",
                dashboard_address=":8787"  # Set this to enable the dashboard
            )
            logger.debug("Dask client initialized.")
            logger.debug(f"Dask dashboard available at {client.dashboard_link}")

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
