import numpy as np
import dask.array as da
from dask import delayed
from dask.distributed import Client

from . import memory

def parallel_compute(
    host_cube,
    valid_pixels,
    compute_pixel,
    multicore=1,
    target_memory_mb=None,
    max_usable_fraction=0.85,
    logger=None
):
    """
    Perform parallel computation using Dask for a generic 3D array.

    This function divides the workload into chunks, processes them in parallel
    using Dask, and stores the computed results in the provided `host_cube`.
    It is designed to handle large datasets efficiently by optimizing memory
    usage and leveraging multi-core processing.

    Parameters
    ----------
    host_cube : numpy.ndarray
        A 3D array that serves as the output container for computed results.
        The shape and data type of this array dictate the computation dimensions
        and memory requirements.
    valid_pixels : list of tuples
        A list of (x, y) pixel coordinates to process. These coordinates specify
        the locations in the 3D array where computations should be performed.
    compute_pixel : callable
        A user-defined function to compute data for a single pixel. The function
        must accept (x, y) as arguments and return a 1D array corresponding to
        the spectral axis of the cube.
    multicore : int, optional
        Number of cores to use for parallel processing. If set to 1, the function
        will not initialize a Dask client and will compute sequentially. Default
        is 1.
    target_memory_mb : int or None, optional
        Desired memory usage per chunk in megabytes. If None, the memory usage
        is dynamically calculated based on the available system memory and the
        `max_usable_fraction`. Default is None.
    max_usable_fraction : float, optional
        The maximum fraction of available system memory to use for computations.
        This helps prevent memory over-allocation. Default is 0.85 (85%).
    logger : logging.Logger or None, optional
        A logger instance for debug and info messages. If None, no logging
        messages will be generated. Default is None.

    Returns
    -------
    numpy.ndarray
        The input `host_cube`, updated with the computed results at the specified
        pixel locations.

    Notes
    -----
    - The function dynamically calculates the chunk size based on the target
      memory and the memory requirements of each spectrum.
    - Dask is used for parallel processing, but if `multicore` is set to 1, the
      computation runs sequentially without initializing a Dask client.
    - This function is designed to efficiently handle large datasets that might
      otherwise exceed system memory when processed all at once.

    Examples
    --------
    Define a function to compute data for a single pixel and use `parallel_compute`:

    >>> import numpy as np
    >>> from mufasa.utils.dask_utils import parallel_compute
    >>> def compute_pixel(x, y):
    ...     return np.array([x + y, x - y])  # Example computation
    >>> host_cube = np.zeros((2, 5, 5))  # A 3D array to store results
    >>> valid_pixels = [(i, j) for i in range(5) for j in range(5)]
    >>> updated_cube = parallel_compute(host_cube, valid_pixels, compute_pixel, multicore=4)
    >>> print(updated_cube.shape)
    (2, 5, 5)
    """


    data_shape = host_cube.shape
    dtype = host_cube.dtype

    if logger:
        logger.info("Running in multi-core mode with Dask.")

    # Chunk computation function
    def compute_chunk(pixel_chunk):
        """Compute data for a chunk of pixels."""
        chunk_data = np.empty((data_shape[0], len(pixel_chunk)), dtype=dtype)
        for i, (x, y) in enumerate(pixel_chunk):
            chunk_data[:, i] = compute_pixel(x, y)
        return chunk_data

    # Calculate target memory if not provided
    if target_memory_mb is None:
        target_memory_mb = memory.calculate_target_memory(multicore, use_total=False,
                                                          max_usable_fraction=max_usable_fraction)
        if logger:
            logger.debug(f"Calculated target memory per chunk: {target_memory_mb:.2f} MB")

    # Calculate memory per spectrum and chunk size
    memory_per_spectrum = data_shape[0] * np.dtype(dtype).itemsize
    chunk_size = max(1, int((target_memory_mb * 1024 * 1024) / memory_per_spectrum))
    chunk_size = min(chunk_size, len(valid_pixels))
    if logger:
        logger.debug(f"Chunk size: {chunk_size}")

    # Split valid pixels into chunks
    pixel_chunks = [valid_pixels[i:i + chunk_size] for i in range(0, len(valid_pixels), chunk_size)]
    if logger:
        logger.debug(f"Number of pixel chunks: {len(pixel_chunks)}")

    # Initialize Dask client if needed
    client = Client(n_workers=multicore) if multicore > 1 else None

    # Create Dask tasks
    results = []
    for chunk in pixel_chunks:
        result = da.from_delayed(
            delayed(compute_chunk)(chunk),
            shape=(data_shape[0], len(chunk)),
            dtype=dtype
        )
        results.append(result)

    # Concatenate results
    if results:
        all_data = da.concatenate(results, axis=1)
        index = 0
        for chunk in pixel_chunks:
            for x, y in chunk:
                host_cube[:, y, x] = all_data[:, index]
                index += 1

    # Close the client if used
    if client:
        client.close()

    return host_cube
