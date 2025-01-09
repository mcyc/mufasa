import numpy as np
import dask.array as da
from pyspeckit.cubes.SpectralCube import Cube

from .utils import dask_utils, memory

#======================================================================================================================#
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#

class PCube(Cube):
    """
    A specialized subclass of :class:`Cube` tailored for Mufasa-specific workflows.

    This class extends the functionality of the :class:`Cube` class to include
    advanced features such as memory-efficient model cube generation and parallelized computation
    using Dask. It is designed to handle large spectral cubes while optimizing for memory usage
    and computational efficiency.
    """

    def get_modelcube(self, update=False, multicore=1, mask=None, target_memory_mb=None,
                      scheduler='threads'):
        """
        Generate or update the model cube using Dask for parallel and memory-efficient processing.

        This method computes a model cube by applying the `specfit.get_full_model` function
        to valid pixels in the cube. It supports both single-core and multi-core processing,
        with optional masking and memory-based chunking.

        Parameters
        ----------
        update : bool, optional
            Whether to force recalculation of the model cube. If False and the model cube
            already exists, the cached version is returned. Default is False.
        multicore : int, optional
            Number of cores to use for parallel processing. If set to 1, computation
            is performed sequentially. Default is 1.
        mask : numpy.ndarray or None, optional
            A 2D boolean mask with the same spatial dimensions as the cube (ny, nx).
            Pixels marked as True in the mask are included in the computation.
            If None, all valid pixels are included. Default is None.
        target_memory_mb : int or None, optional
            Target memory usage per chunk in megabytes. If None, the memory limit is
            estimated based on system resources. Default is None.
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
            A Dask array representing the computed model cube. Pixels that were not
            included in the computation (due to masking or invalid data) are set to NaN.

        Notes
        -----
        - The computation is performed pixel-by-pixel, preserving the spectral axis in full.
        - For single-core computation, the method calls `lazy_pix_compute` to process each valid pixel sequentially.
          See :func:`mufasa.utils.dask_utils.lazy_pix_compute`.
        - For multi-core computation, the method calls `lazy_pix_compute_multiprocessing`, which uses Dask's
          parallel computing capabilities with batch processing for efficiency. See
          :func:`mufasa.utils.dask_utils.lazy_pix_compute_multiprocessing`.
        - The number of cores is adjusted based on the number of valid pixels to avoid unnecessary resource allocation.
        - The memory per worker is limited by the system's available resources, and batch processing ensures
          memory-efficient computation. Chunk sizes are calculated dynamically using
          :func:`mufasa.utils.dask_utils.calculate_chunks`.
        - The optimal batch size for multi-core processing is determined using
          :func:`mufasa.utils.dask_utils.calculate_batch_size`.

        Example
        -------
        >>> cube = MySpectralCube(...)  # User-defined subclass of SpectralCube
        >>> model_cube = cube.get_modelcube(update=True, multicore=4)
        >>> print(model_cube.shape)
        (100, 200, 200)  # Example dimensions
        """
        if self._modelcube is not None and not update:
            return self._modelcube

        nanvals = ~np.all(np.isfinite(self.parcube), axis=0)
        isvalid = np.any(self.parcube, axis=0) & ~nanvals

        if mask is not None:
            isvalid &= mask

        n_pix = np.sum(isvalid)

        if n_pix < 1:
            logger.warning("No valid pixels to update the model.")
            return self._modelcube

        if multicore > n_pix:
            # ensure we're not requesting more cores than pixels
            multicore = n_pix

        if self._modelcube is None:
            # Calculate chunks
            chunks = dask_utils.calculate_chunks(cube_shape=self.cube.shape, dtype=self.cube.dtype, target_chunk_mem_mb=128)
            # initializing modelcube
            self._modelcube = da.full(shape=self.cube.shape, fill_value=np.nan, dtype=self.cube.dtype,
                                chunks=chunks)

        def compute_pixel(x, y):
            """Compute the model spectrum for a single pixel."""
            return self.specfit.get_full_model(pars=self.parcube[:, y, x])

        if multicore == 1:
            # Sequential computation
            logger.info("Running in single-core mode.")
            self._modelcube = dask_utils.lazy_pix_compute_single(self._modelcube, isvalid, compute_pixel)

        else:
            # Parallel computation with Dask when justified

            # estimate the memory available
            memory_max = 2048 # 2 GB per worker
            target_memory_mb = memory.calculate_target_memory(multicore, use_total=False, max_usable_fraction=0.85)
            if target_memory_mb > memory_max:
                target_memory_mb = memory_max

            # calculate the optimal batch_size and cores
            batch_size, multicore = dask_utils.calculate_batch_size(
                spectral_size=self._modelcube.shape[0],
                dtype=self._modelcube.dtype,
                total_valid_pixels=n_pix,
                n_cores=multicore,
                memory_limit_mb=target_memory_mb,
                task_per_core=200,
                min_tpc=5
            )

            if multicore > 1 and batch_size > 0:
                logger.debug("Computing model in multi-core mode with Dask.")
                logger.debug(f"batch size: {batch_size}")
                logger.debug(f"n cores: {multicore}")

                self._modelcube = dask_utils.lazy_pix_compute(self._modelcube, isvalid, compute_pixel,
                                              batch_size=batch_size, n_workers=multicore, scheduler=scheduler)

            else:
                logger.debug("Computing model in single-core mode.")
                self._modelcube = dask_utils.lazy_pix_compute(self._modelcube, isvalid, compute_pixel)

        return self._modelcube