"""
This module provides general purpose tool and wrappers to handle dask object.
Some of the key methods support parallel computing with effecient memory usage.
"""
import gc
import numpy as np
import dask.array as da
import psutil
import shutil
import functools
from dask import delayed, config, base
from dask.distributed import Client, LocalCluster, get_client, TimeoutError

# ======================================================================================================================#
from .mufasa_log import get_logger

logger = get_logger(__name__)
#======================================================================================================================#

# for monitoring

from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler, visualize
from functools import wraps

def profile_and_visualize(**visualize_kwargs):
    """
    A decorator to profile the execution of a function using Dask's profiling tools
    and visualize the results.

    The decorator uses `Profiler`, `ResourceProfiler`, and `CacheProfiler` from
    Dask to collect performance metrics during the function execution. After
    the function completes, the collected data is visualized using the
    `visualize` function.

    Parameters
    ----------
    **visualize_kwargs : dict
        Additional keyword arguments to be passed to the `visualize` function.
        These can include options such as `color`, `filename`, or other
        visualization settings.

    Returns
    -------
    function
        The wrapped function with profiling and visualization enabled.

    Examples
    --------
    >>> @profile_and_visualize(color="blue", save="profile.html")
    ... def convolve():
    ...     # Perform computationally intensive operations
    ...     pass
    ...
    >>> convolve()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
                result = func(*args, **kwargs)  # Call the original function
            visualize([prof, rprof, cprof], **visualize_kwargs)  # Visualize the profiling data
            return result  # Return the result of the original function
        return wrapper
    return decorator


def ensure_dask_client(func):
    """
    Decorator to ensure a Dask Client is available when the wrapped function is called,
    but only if the `local_cluster_kwargs` keyword argument is provided.

    Ensures a Dask client is running. If no client exists, it creates one using LocalCluster.
    Returns a client instance and a boolean indicating whether the client was created locally.

    the for LocalCluster can be found at https://docs.dask.org/en/latest/deploying-python.html
    name=None, n_workers=None, threads_per_worker=None, processes=None, loop=None,
    start=None, host=None, ip=None, scheduler_port=0, silence_logs=30, dashboard_address=':8787',
    worker_dashboard_address=None, diagnostics_port=None, services=None, worker_services=None,
    service_kwargs=None, asynchronous=False, security=None, protocol=None, blocked_handlers=None,
    interface=None, worker_class=None, scheduler_kwargs=None, scheduler_sync_interval=1, **worker_kwargs

    Parameters
    ----------
    func : callable
        The function to be decorated.

    Returns
    -------
    callable
        The wrapped function with Dask client handling.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the `local_cluster_kwargs` from the kwargs, if present

        local_cluster_kwargs = kwargs.pop('local_cluster_kwargs', None)
        client = None
        created_locally = False

        try:
            client = get_client()  # Check for an existing client
        except (ValueError, TimeoutError):
            # No client found, so create a local cluster and client

            if local_cluster_kwargs is not None:
                # Create a LocalCluster and Client only if `local_cluster_kwargs` is specified
                if isinstance(local_cluster_kwargs, dict):
                    cluster = LocalCluster(**local_cluster_kwargs)
                else:
                    cluster = LocalCluster()
                client = Client(cluster)
                created_locally = True

        try:
            # Call the wrapped function
            return func(*args, **kwargs)
        finally:
            # If we created the client locally, clean it up
            if created_locally and client is not None:
                client.close()

    return wrapper

def set_dask_config(func):
    """
    Set dask.config.scheduler (global) for the function and return it to the previous state
    When the function returns. Currently only take scheduler
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the `local_cluster_kwargs` from the kwargs, if present

        if 'scheduler' in kwargs:
            scheduler = kwargs['scheduler']
        else:
            scheduler = None

        try:
            scheduler_og = config.get('scheduler')
        except KeyError:
            scheduler_og = base.get_scheduler()

        if scheduler:
            config.set(scheduler=scheduler)

        try:
            # Call the wrapped function
            return func(*args, **kwargs)
        finally:
            config.set(scheduler=scheduler_og)

    return wrapper

#======================================================================================================================#


def store_to_zarr(darray, filename, readback=False, delete_old=True):
    # compute it onto a disk temporary and retrive it
    if delete_old:
        try:
            shutil.rmtree(filename)
        except FileNotFoundError:
            pass
    darray.to_zarr(filename, overwrite=True, compute=True)

    if readback:
        return da.from_zarr(filename)

def dask_array_stats(array):
    """
    Quickly analyze a Dask array's chunk structure and memory requirements.

    Parameters
    ----------
    array : dask.array.Array
        The Dask array to analyze.

    Returns
    -------
    dict
        A dictionary containing the number of chunks in each dimension,
        the total number of chunks, and an estimate of total memory.
    """
    # Number of chunks in each dimension
    chunks_per_dimension = [len(dim) for dim in array.chunks]

    # Total number of chunks (product of chunks along dimensions)
    total_chunks = np.prod(chunks_per_dimension)

    # Estimate total memory
    total_memory = array.nbytes  # Total bytes in the array (dtype * size of array)

    return total_chunks, total_memory / 1e6,  # Convert to MB



def reset_graph(dask_obj):
    """
    Resets the computation graph of a Dask object.

    Parameters:
    ----------
    dask_obj : dask.array.Array, dask.dataframe.DataFrame, or dask.bag.Bag
        The Dask object to reset.

    Returns:
    -------
    Same type as input
        A new Dask object with the computation graph reset.
    """
    if hasattr(dask_obj, "map_blocks"):
        # Handle Dask arrays
        dask_cleaned = dask_obj.map_blocks(lambda block: block, dtype=dask_obj.dtype)
    elif hasattr(dask_obj, "map_partitions"):
        # Handle Dask DataFrames or Bags
        dask_cleaned = dask_obj.map_partitions(lambda partition: partition)
    else:
        raise TypeError("Unsupported Dask object type.")

    gc.collect()  # Optional: Trigger garbage collection
    return dask_cleaned


def persist_and_clean(dask_obj, debug=False, visualize_filename=None):
    """
    Persist a Dask collection and optionally visualize its computation graph.

    Parameters
    ----------
    dask_obj : dask.array.Array or dask.dataframe.DataFrame or dask.bag.Bag
        The Dask collection to persist.
    debug : bool, optional, default=False
        If True, visualize the computation graph.
    visualize_filename : str or None, optional, default=None
        File name to save the visualization of the computation graph.
        If None, no file is saved.

    Returns
    -------
    dask_obj
        The persisted Dask collection.

    Examples
    --------
    Persist a Dask array and visualize its graph:

    >>> import dask.array as da
    >>> x = da.random.random((1000, 1000), chunks=(100, 100))
    >>> x_persisted = persist_and_clean(x, debug=True, visualize_filename="graph.png")
    >>> print(x_persisted)
    """
    persisted_result = dask_obj.persist()
    del dask_obj
    gc.collect()

    if debug and visualize_filename:
        persisted_result.visualize(filename=visualize_filename)

    return persisted_result


def chunk_by_slice(cube_shape, dtype, target_chunk_mb=128):
    """
    Calculate Dask chunking into 3D image slices, keeping planes as intact as possible

    By not breaking up the image should make image plane convolution more effecient

    Parameters
    ----------
    cube_shape : tuple of int
        The shape of the data cube in the form (spectral_axis, y, x).
    dtype : numpy.dtype
        The data type of the data cube (e.g., `numpy.float32`, `numpy.float64`).
    target_chunk_mb : int, optional, default=128
        The target memory size per chunk, in megabytes.

    Returns
    -------
    tuple of int
        Optimal chunk sizes for the Dask
    """
    z, y, x = cube_shape

    # Convert target MB to bytes
    target_bytes = target_chunk_mb * 1024**2

    # Size of one element in bytes
    element_size = np.dtype(dtype).itemsize

    # Initial guess for chunk sizes in y and x
    chunk_y = y
    chunk_x = x

    # Determine the largest chunks for y and x within the target size
    while chunk_y * chunk_x * element_size > target_bytes:
        if chunk_y >= chunk_x:
            chunk_y //= 2
        else:
            chunk_x //= 2

    # Calculate the remaining chunk size for z (i.e., the spectral dimension)
    max_elements_per_chunk = target_bytes // (chunk_y * chunk_x * element_size)
    chunk_z = max(1, min(z, max_elements_per_chunk))

    # Ensure y and x are evenly divisible by chunk sizes
    chunk_y = min(y, (y // chunk_y) * chunk_y)
    chunk_x = min(x, (x // chunk_x) * chunk_x)

    # Finalize the chunking scheme
    chunks = (chunk_z, chunk_y, chunk_x)
    return chunks

# Example usage:



def chunk_by_ray(cube_shape, dtype, target_chunk_mb=128):
    """
    Calculate chunk sizes for a Dask array with an optimal aspect ratio and target memory usage.

    This function computes chunk sizes for a data cube that approximate square chunks
    (lowest aspect ratio difference) while adhering to a target memory limit per chunk.

    Parameters
    ----------
    cube_shape : tuple of int
        The shape of the data cube in the form (spectral_axis, y, x).
    dtype : numpy.dtype
        The data type of the data cube (e.g., `numpy.float32`, `numpy.float64`).
    target_chunk_mb : int, optional, default=128
        The target memory size per chunk, in megabytes.

    Returns
    -------
    tuple of int
        Optimal chunk sizes for the Dask array in the form
        (spectral_chunk, y_chunk, x_chunk).

    Notes
    -----
    - The function prioritizes chunks with a low aspect ratio to maintain
      computational efficiency.
    - The spectral axis is kept in a single chunk.

    Examples
    --------
    Calculate chunks for a data cube with a shape of (100, 1000, 1000)
    and a `float32` data type:

    >>> chunk_by_ray((100, 1000, 1000), np.float32, target_chunk_mb=64)
    (100, 250, 250)

    Calculate chunks with a higher memory limit:

    >>> chunk_by_ray((100, 1000, 1000), np.float32, target_chunk_mb=512)
    (100, 500, 500)
    """
    spectral_axis, y_dim, x_dim = cube_shape
    dtype_size = np.dtype(dtype).itemsize  # Size of one element in bytes

    # Convert target memory from MB to bytes
    target_chunk_mem_bytes = target_chunk_mb * 1024 * 1024

    # Total memory per pixel across the spectral axis
    memory_per_pixel = spectral_axis * dtype_size

    # Number of spatial pixels that fit in the target memory
    spatial_pixels_per_chunk = target_chunk_mem_bytes // memory_per_pixel

    # Initialize best chunk sizes and aspect ratio difference
    best_y_chunk = y_dim
    best_x_chunk = x_dim
    best_aspect_ratio_diff = float("inf")

    # Iterate over possible chunk sizes to find the closest to square
    for y_chunk in range(1, y_dim + 1):
        x_chunk = spatial_pixels_per_chunk // y_chunk
        if x_chunk > x_dim:
            continue

        # Calculate aspect ratio difference
        aspect_ratio_diff = abs(y_chunk - x_chunk)

        # Update best chunk sizes if a better aspect ratio is found
        if aspect_ratio_diff < best_aspect_ratio_diff:
            best_y_chunk = y_chunk
            best_x_chunk = x_chunk
            best_aspect_ratio_diff = aspect_ratio_diff

        # If perfect square chunks are found, stop searching
        if aspect_ratio_diff == 0:
            break

    # Use the entire spectral axis in one chunk
    spectral_chunk = spectral_axis

    # Ensure there are close to integer number of chunks in the data
    n_y, n_x = np.rint(y_dim / best_y_chunk), np.rint(x_dim / best_x_chunk)
    best_y_chunk = np.ceil(y_dim / n_y).astype(int)
    best_x_chunk = np.ceil(x_dim / n_x).astype(int)

    return (spectral_chunk, best_y_chunk, best_x_chunk)


def calculate_batch_size(spectral_size, dtype, total_valid_pixels, n_cores, memory_limit_mb=1024,
                         task_per_core=10, min_tpc=5):
    """
    Calculate the optimal batch size and adjusted number of workers for processing valid pixels.

    This function computes an optimal batch size for processing data, balancing memory constraints
    and task distribution across available workers. It ensures efficient parallelism while respecting
    memory limits and task-per-core requirements.

    Parameters
    ----------
    spectral_size : int
        Length of the spectral axis (z-dimension) of the data cube.
    dtype : numpy.dtype
        Data type of the cube (e.g., `numpy.float32`, `numpy.float64`) used to calculate memory usage per pixel.
    total_valid_pixels : int
        Total number of valid pixels to process. Typically defined by a mask or selection criteria.
    n_cores : int
        Number of workers available for parallel processing.
    memory_limit_mb : int, optional, default=1024
        Approximate memory limit per worker in megabytes.
    task_per_core : int, optional, default=10
        Target number of tasks per core for efficient parallelism.
    min_tpc : int, optional, default=5
        Minimum number of tasks per core to ensure effective task distribution.

    Returns
    -------
    tuple of int
        - Optimal batch size for processing valid pixels.
        - Adjusted number of workers to balance workload and task distribution.

    Notes
    -----
    - The batch size is constrained by the memory limit per worker and the total valid pixels.
    - If the total number of tasks does not meet the minimum tasks-per-core requirement, the function adjusts
      the number of workers or signals that the workload needs adjustment by setting the batch size to 0.
    - The function prioritizes evenly distributed tasks while respecting memory constraints.

    Examples
    --------
    Compute the batch size and adjusted number of workers for a data cube:

    >>> spectral_size = 1000
    >>> dtype = np.float32
    >>> total_valid_pixels = 10000
    >>> n_cores = 4
    >>> calculate_batch_size(spectral_size, dtype, total_valid_pixels, n_cores)
    (250, 4)

    Adjust the memory limit and tasks-per-core:

    >>> calculate_batch_size(1000, np.float32, 20000, 8, memory_limit_mb=512, task_per_core=20)
    (125, 8)
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


def compute_global_offsets(chunks, chunk_location):
    """
    Calculate global offsets for a chunk's position in a multidimensional array.

    Parameters
    ----------
    chunks : tuple of tuple of int
        Chunk sizes for each dimension. For example, `((4, 4, 4), (6, 6), (5, 5, 5, 5))`.
    chunk_location : tuple of int
        Indices of the chunk along each dimension. For example, `(1, 0, 2)`.

    Returns
    -------
    tuple of int
        Global offsets for the chunk along each dimension.

    Examples
    --------
    >>> chunks = ((4, 4, 4), (6, 6), (5, 5, 5, 5))
    >>> chunk_location = (1, 0, 2)
    >>> compute_global_offsets(chunks, chunk_location)
    (4, 0, 10)
    """
    offsets = []
    for dim_chunks, loc in zip(chunks, chunk_location):
        offset = sum(dim_chunks[:loc])
        offsets.append(offset)
    return tuple(offsets)


def lazy_pix_compute_no_batching(host_cube, isvalid, compute_pixel, scheduler='threads', use_global_xy=True):
    """
    Compute valid pixels in a Dask array by processing only relevant chunks.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array with dimensions (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D mask where `True` indicates valid pixels to process.
    compute_pixel : callable
        Function that computes spectral values for a given pixel `(x, y)`.
        The signature should be `compute_pixel(global_x, global_y)`.
    scheduler : {'threads', 'processes', 'synchronous'}, optional, default='threads'
        Scheduler to use for Dask computation.
    use_global_xy : bool, optional, default=True
        If `True`, passes global coordinates `(global_x, global_y)` to `compute_pixel`.
        If `False`, uses local chunk coordinates.

    Returns
    -------
    dask.array.Array
        A Dask array with updated values for valid pixels.

    Notes
    -----
    - Only relevant chunks containing valid pixels are processed to optimize performance.
    - If no valid pixels are present in a chunk, the chunk is returned unchanged.

    Examples
    --------
    Compute valid pixels for a Dask array:

    >>> import dask.array as da
    >>> import numpy as np
    >>> host_cube = da.random.random((10, 100, 100), chunks=(5, 50, 50))
    >>> isvalid = np.random.choice([True, False], size=(100, 100), p=[0.1, 0.9])
    >>> def compute_pixel(x, y):
    ...     return np.ones(10) * (x + y)
    >>> result = lazy_pix_compute_no_batching(host_cube, isvalid, compute_pixel)
    >>> result.compute()
    """
    # Ensure `isvalid` is a Dask array with proper chunks
    _, y_chunks, x_chunks = host_cube.chunks
    if not isinstance(isvalid, da.Array):
        isvalid = da.from_array(isvalid, chunks=(y_chunks, x_chunks))
    elif isvalid.chunks != (y_chunks, x_chunks):
        isvalid = isvalid.rechunk((y_chunks, x_chunks))

    def update_chunk(chunk, isvalid_chunk, block_info=None):
        # Skip processing if this chunk is not relevant

        # Find valid pixel indices in this chunk
        valid_pixel_indices = np.argwhere(isvalid_chunk)

        if len(valid_pixel_indices) < 1:
            # return the chunk if there's no valid pixel in the chunk
            return chunk

        chunk_location = block_info[0]['chunk-location']
        chunk_offset_y, chunk_offset_x = compute_global_offsets((y_chunks, x_chunks), chunk_location[1:])

        chunk_copy = np.copy(chunk)
        for local_y, local_x in valid_pixel_indices:
            global_y, global_x = (local_y + chunk_offset_y, local_x + chunk_offset_x) if use_global_xy else (local_y, local_x)
            chunk_copy[:, local_y, local_x] = compute_pixel(global_x, global_y)
        return chunk_copy

    # Apply the function using map_blocks
    with config.set(scheduler=scheduler):
        logger.debug(f"Scheduler: {scheduler}")
        result = da.map_blocks(
            update_chunk,
            host_cube,
            isvalid,
            dtype=host_cube.dtype,
        )
        return persist_and_clean(result, debug=False, visualize_filename=None)


def compute_chunk_relevant(isvalid):
    """
    Determine the relevance of chunks in a 2D Boolean mask.

    For each chunk in the 2D Dask array, checks whether the chunk contains any
    `True` values. Outputs a 2D Boolean array indicating which chunks are relevant.

    Parameters
    ----------
    isvalid : dask.array.Array
        A 2D Boolean mask with spatial chunking.

    Returns
    -------
    numpy.ndarray
        A 2D Boolean array where each element indicates whether the corresponding
        chunk in the input mask contains any `True` values.

    Examples
    --------
    >>> import dask.array as da
    >>> import numpy as np
    >>> isvalid = da.from_array(np.array([[True, False], [False, False]]), chunks=(1, 2))
    >>> compute_chunk_relevant(isvalid)
    array([[ True],
           [False]])
    """
    chunk_relevant = isvalid.map_blocks(
        lambda block: np.array([[block.any()]]),  # Reduce block and wrap as 2D
        dtype=bool,
    ).compute()

    return chunk_relevant


def custom_task_graph_pixel(host_cube, isvalid, compute_pixel, use_global_xy=True, scheduler='threads'):
    """
    Construct a custom task graph to process valid pixels in a Dask array.

    This function processes only the relevant chunks of the input 3D Dask array, updating
    pixel values based on a provided function and a 2D validity mask.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array with dimensions (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D mask where `True` indicates valid pixels to process.
    compute_pixel : callable
        A function to compute spectral values for a given pixel `(x, y)`. The function signature
        should be `compute_pixel(global_x, global_y)`.
    use_global_xy : bool, optional, default=True
        If `True`, passes global coordinates `(global_x, global_y)` to `compute_pixel`.
        If `False`, uses local chunk coordinates.
    scheduler : {'threads', 'processes', 'synchronous'}, optional, default='threads'
        The Dask scheduler to use for computation.

    Returns
    -------
    dask.array.Array
        A 3D Dask array with updated values for valid pixels.

    Notes
    -----
    - Non-relevant chunks (those with no valid pixels) are returned unchanged.
    - Relevant chunks are processed to update pixel values using the provided `compute_pixel` function.
    - The function leverages Dask's `da.block` and delayed tasks to construct the final Dask array.

    Examples
    --------
    Process a Dask array with a custom task graph:

    >>> import dask.array as da
    >>> import numpy as np
    >>> host_cube = da.random.random((10, 100, 100), chunks=(5, 50, 50))
    >>> isvalid = np.random.choice([True, False], size=(100, 100), p=[0.1, 0.9])
    >>> def compute_pixel(x, y):
    ...     return np.ones(10) * (x + y)
    >>> result = custom_task_graph_pixel(host_cube, isvalid, compute_pixel, scheduler='threads')
    >>> result.compute()
    """
    # Ensure `isvalid` is a Dask array with proper chunks
    _, y_chunks, x_chunks = host_cube.chunks
    if not isinstance(isvalid, da.Array):
        isvalid = da.from_array(isvalid, chunks=(y_chunks, x_chunks))
    elif isvalid.chunks != (y_chunks, x_chunks):
        isvalid = isvalid.rechunk((y_chunks, x_chunks))

    # Compute chunk relevance
    try:
        chunk_relevant = compute_chunk_relevant(isvalid)
    except Exception as e:
        raise RuntimeError(f"Error computing chunk relevance: {e}")

    # Convert to delayed arrays
    host_cube_delayed = host_cube.to_delayed().tolist()
    isvalid_delayed = isvalid.to_delayed().tolist()

    # Create Dask arrays for each chunk
    dask_arrays_reshaped = []
    for chunk_idx, relevant in np.ndenumerate(chunk_relevant):
        spatial_idx = chunk_idx

        host_chunk = host_cube_delayed[0][spatial_idx[0]][spatial_idx[1]]
        isvalid_chunk = isvalid_delayed[spatial_idx[0]][spatial_idx[1]]

        if relevant:
            @delayed
            def process_chunk(host_chunk, isvalid_chunk, spatial_idx):
                valid_pixel_indices = np.argwhere(isvalid_chunk)

                host_chunk = host_chunk.copy()
                # Compute global offsets
                chunk_location = (spatial_idx[0], spatial_idx[1])
                chunk_offset_y, chunk_offset_x = compute_global_offsets((y_chunks, x_chunks), chunk_location)

                for local_y, local_x in valid_pixel_indices:
                    global_y, global_x = (local_y + chunk_offset_y, local_x + chunk_offset_x) if use_global_xy else (local_y, local_x)
                    host_chunk[:, local_y, local_x] = compute_pixel(global_x, global_y)
                return host_chunk

            delayed_task = process_chunk(host_chunk, isvalid_chunk, spatial_idx)
        else:
            # Wrap non-relevant chunks in delayed
            delayed_task = delayed(lambda x: x)(host_chunk)

        # Convert the delayed task into a Dask array
        dask_array = da.from_delayed(
            delayed_task,
            shape=(host_cube.chunks[0][0], y_chunks[spatial_idx[0]], x_chunks[spatial_idx[1]]),
            dtype=host_cube.dtype,
        )
        if len(dask_arrays_reshaped) <= spatial_idx[0]:
            dask_arrays_reshaped.append([])
        dask_arrays_reshaped[spatial_idx[0]].append(dask_array)

    # Use da.block to combine the Dask arrays
    try:
        result = da.block(dask_arrays_reshaped)
    except Exception as e:
        raise RuntimeError(f"Error constructing Dask array from Dask arrays: {e}")

    # Apply the scheduler
    with config.set(scheduler=scheduler):
        logger.debug(f"Using scheduler: {scheduler}")
        return persist_and_clean(result, debug=False, visualize_filename=None)


def lazy_pix_compute_single(host_cube, isvalid, compute_pixel):
    """
    Lazily compute spectral values for valid pixels in a data cube.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array representing the data cube with dimensions (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D Boolean mask of shape (y, x). Pixels with `True` are computed, and others remain `NaN`.
    compute_pixel : callable
        A function that computes spectral values for a pixel. It should accept `(x, y)` as inputs
        and return a 1D array of spectral values.

    Returns
    -------
    dask.array.Array
        A 3D Dask array with the same shape as `host_cube`, where computed values replace `NaN`
        for valid pixels.

    Notes
    -----
    - Pixels not marked as valid (`False` in `isvalid`) remain as `NaN` in the output array.
    - The computation for each pixel is performed lazily using Dask's delayed framework.
    - This function assumes `isvalid` has the same spatial dimensions as `host_cube`.

    Examples
    --------
    Compute spectral values for a subset of pixels in a Dask array:

    >>> import dask.array as da
    >>> import numpy as np
    >>> host_cube = da.random.random((10, 100, 100), chunks=(5, 50, 50))
    >>> isvalid = np.random.choice([True, False], size=(100, 100), p=[0.1, 0.9])
    >>> def compute_pixel(x, y):
    ...     return np.ones(10) * (x + y)
    >>> result = lazy_pix_compute_single(host_cube, isvalid, compute_pixel)
    >>> result.compute()
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
    Compute valid pixels in a 3D data cube adaptively with dynamic batching and scheduling.

    This function dynamically determines the optimal batch size and scheduler based on memory limits,
    CPU cores, and valid pixel density. It processes only the relevant pixels in the data cube lazily.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array representing the data cube with dimensions (spectral, y, x).
    isvalid : numpy.ndarray or dask.array.Array
        A 2D Boolean mask of shape (y, x). Pixels with `True` are computed, while others remain `NaN`.
    compute_pixel : callable
        A function that computes spectral values for a pixel. It should accept `(x, y)` as inputs
        and return a 1D array of spectral values.
    memory_limit_mb : int, optional, default=1024
        Memory limit per worker in megabytes. Used to determine batch size.
    scheduler : {'adaptive', 'threads', 'processes', 'synchronous'}, optional, default='adaptive'
        The scheduler to use for Dask computation. If 'adaptive', selects 'threads' for high-density
        masks and 'processes' for low-density masks.

    Returns
    -------
    dask.array.Array
        A 3D Dask array with the same shape as `host_cube`, where computed values replace `NaN`
        for valid pixels.

    Notes
    -----
    - The batch size is determined dynamically based on memory constraints and valid pixel density.
    - The scheduler is chosen adaptively if 'adaptive' is specified, prioritizing thread-based or
      process-based execution depending on mask density.
    - Pixels not marked as valid remain `NaN` in the output array.

    Examples
    --------
    Compute valid pixels adaptively for a Dask array:

    >>> import dask.array as da
    >>> import numpy as np
    >>> host_cube = da.random.random((10, 100, 100), chunks=(5, 50, 50))
    >>> isvalid = np.random.choice([True, False], size=(100, 100), p=[0.1, 0.9])
    >>> def compute_pixel(x, y):
    ...     return np.ones(10) * (x + y)
    >>> result = lazy_pix_compute_dynamic(host_cube, isvalid, compute_pixel, memory_limit_mb=512, scheduler='adaptive')
    >>> result.compute()
    """
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
    Lazily compute spectral values for valid pixels in a data cube.

    This function processes only the pixels marked as `True` in the `isvalid` mask,
    using the provided `compute_pixel` function. It supports parallel computation
    through Dask's schedulers or a distributed client for batching.

    Parameters
    ----------
    host_cube : dask.array.Array
        A 3D Dask array with dimensions (spectral, y, x), representing the data cube.
    isvalid : numpy.ndarray or dask.array.Array
        A 2D Boolean mask of shape (y, x). Pixels marked as `True` are processed;
        others remain `NaN`.
    compute_pixel : callable
        A function that computes spectral values for a pixel. It must accept two arguments,
        `(x, y)`, and return a 1D array of spectral values.
    batch_size : int, optional, default=100
        Number of pixels to process in each batch. Larger batches reduce scheduling
        overhead but require more memory.
    n_workers : int, optional, default=4
        Number of workers for parallel processing. Used only with the `'batch'` scheduler.
    scheduler : {'threads', 'processes', 'synchronous', 'batch'}, optional, default='threads'
        The scheduler to use for computation:
        - `'threads'`: Thread-based execution with shared memory.
        - `'processes'`: Process-based execution with separate memory spaces.
        - `'synchronous'`: Single-threaded execution.
        - `'batch'`: Uses a Dask distributed client for batching and access to a dashboard.

    Returns
    -------
    dask.array.Array
        A 3D Dask array with computed values for valid pixels. Non-valid pixels remain `NaN`.

    Notes
    -----
    - When using the `'batch'` scheduler, a local Dask distributed client is initialized
      for batch execution and dashboard monitoring.
    - Ensure that `compute_pixel` is thread-safe if using the `'threads'` scheduler.
    - Batching is suitable for large datasets or dense masks, as it minimizes overhead.

    Examples
    --------
    Compute spectral values for valid pixels in a data cube:

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
    >>> result.compute()  # Trigger computation and retrieve the result.
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
    Update a single pixel in a block of a Dask array.

    This helper function modifies a copy of the block by assigning the provided
    spectral value to the specified pixel coordinates.

    Parameters
    ----------
    block : np.ndarray
        A block from the Dask array being processed. Typically a 3D array
        with dimensions (spectral, y, x).
    value : np.ndarray
        A 1D array representing the computed spectral values for the pixel.
    x : int
        The x-coordinate of the pixel within the block.
    y : int
        The y-coordinate of the pixel within the block.

    Returns
    -------
    np.ndarray
        A copy of the block with the specified pixel updated with the new value.

    Notes
    -----
    - The function creates a copy of the input block to ensure that updates
      do not modify the original array in place.
    - Coordinates `(x, y)` are relative to the block, not global coordinates.
    """
    block_copy = block.copy()
    block_copy[:, y, x] = value
    return block_copy


def _update_batch(block, batch_result):
    """
    Update multiple pixels in a block of a Dask array using batch results.

    This function modifies a copy of the block by applying computed spectral values
    for a batch of pixels.

    Parameters
    ----------
    block : numpy.ndarray
        A 3D block from the Dask array being processed. The block has dimensions
        (spectral_axis, y, x).
    batch_result : tuple
        A tuple containing:
        - batch_result[0]: list of tuple of int
            A list of `(y, x)` pixel coordinates for the batch.
        - batch_result[1]: numpy.ndarray
            A 2D array of computed spectral values for the batch. Shape:
            `(spectral_axis, num_pixels)`.

    Returns
    -------
    numpy.ndarray
        A copy of the input block with the specified pixels updated based on the
        batch results.

    Notes
    -----
    - The function creates a copy of the input block to ensure the original block
      remains unmodified.
    - Coordinates `(y, x)` in `batch_result[0]` are relative to the block, not
      global coordinates.

    Examples
    --------
    Update a block with batch results:

    >>> block = np.zeros((5, 4, 4))  # (spectral_axis, y, x)
    >>> batch_result = (
    ...     [(0, 1), (2, 3)],  # Pixel coordinates
    ...     np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])  # Spectral values
    ... )
    >>> updated_block = _update_batch(block, batch_result)
    >>> updated_block[:, 0, 1]  # Spectral values for pixel (0, 1)
    array([ 1.,  3.,  5.,  7.,  9.])
    >>> updated_block[:, 2, 3]  # Spectral values for pixel (2, 3)
    array([ 2.,  4.,  6.,  8., 10.])
    """
    block_copy = block.copy()  # Create a writable copy
    batch_pixels, batch_data = batch_result

    for i, (y, x) in enumerate(batch_pixels):
        block_copy[:, y, x] = batch_data[:, i]  # Assign spectral values to each pixel

    # Clean up variables to free memory
    del block
    del batch_result
    del batch_pixels, batch_data

    return block_copy


def _is_valid_chunk(chunks):
    """
    Validate the compatibility of the `chunks` argument with Dask.

    Parameters
    ----------
    chunks : None, dict, tuple, or list
        The chunking strategy to validate. Valid inputs are:
        - None: Indicates no chunking is specified.
        - dict: A dictionary where keys are strings (dimension names) and values are integers or tuples of integers.
        - tuple or list: A sequence of integers or tuples of integers specifying chunk sizes for each dimension.

    Returns
    -------
    bool
        True if `chunks` is valid.

    Raises
    ------
    ValueError
        If `chunks` is not None, a dictionary, a tuple, or a list, or if the contents of the dictionary or sequence
        do not adhere to the expected structure.
    """
    if chunks is None:
        return True  # None is acceptable for chunks

    if isinstance(chunks, dict):
        # Ensure all keys are strings and values are integers or tuples of integers
        if not all(isinstance(key, str) for key in chunks):
            raise ValueError("When `chunks` is a dictionary, all keys must be strings.")
        if not all(
                isinstance(value, (int, tuple)) and
                (isinstance(value, tuple) and all(isinstance(v, int) for v in value) or isinstance(value, int))
                for value in chunks.values()
        ):
            raise ValueError("When `chunks` is a dictionary, all values must be integers or tuples of integers.")
    elif isinstance(chunks, (tuple, list)):
        # Ensure all elements are integers or tuples of integers
        if not all(
                isinstance(c, (int, tuple)) and
                (isinstance(c, tuple) and all(isinstance(v, int) for v in c) or isinstance(c, int))
                for c in chunks
        ):
            raise ValueError("When `chunks` is a list or tuple, all elements must be integers or tuples of integers.")
    else:
        raise ValueError("`chunks` must be None, a dictionary, or a tuple/list of integers.")

    return True
