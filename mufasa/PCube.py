"""
The `mufasa.PCube` module provides dask-centered subclass of `pyspeckit`'s :class:`Cube`
that features some enhancing class methods tailored for MUFASA

"""

import numpy as np
import dask.array as da
from pyspeckit.cubes.SpectralCube import Cube
from copy import deepcopy

from .utils import dask_utils
from .utils.multicore import validate_n_cores

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

    def get_modelcube(self, update=False, multicore=True, mask=None, target_memory_mb=16, scheduler='threads'):
        """
        Generate or update the model cube using parallel, memory-efficient processing with Dask.

        This method computes the model cube by applying the `specfit.get_full_model` function
        to valid pixels in the data cube. It supports both single-core and multi-core processing,
        with optional masking and memory-based chunking.

        Parameters
        ----------
        update : bool, optional, default=False
            If `True`, forces recalculation of the model cube. If `False` and a model cube
            already exists, returns the cached version.
        multicore : bool or int, optional, default=True
            Enables multi-core processing if `True`. If an integer is provided, it specifies
            the number of cores to use. If `False`, computation is performed in single-core mode.
        mask : numpy.ndarray or None, optional, default=None
            A 2D Boolean mask of shape `(ny, nx)`. Pixels marked as `True` are included in
            the computation. If `None`, all valid pixels are processed.
        target_memory_mb : int, optional, default=16
            The target memory usage per chunk in megabytes. If `None`, memory is estimated
            based on available system resources.
        scheduler : {'threads', 'processes', 'synchronous', 'batch'}, optional, default='threads'
            The Dask scheduler for parallel computation:
            - `'threads'`: Use thread-based parallelism (default).
            - `'processes'`: Use process-based parallelism.
            - `'synchronous'`: Perform computations sequentially.
            - `'batch'`: Use Dask's distributed client for batching and dashboard monitoring.

        Returns
        -------
        dask.array.Array
            A Dask array representing the computed model cube. Non-valid pixels are set to `NaN`.

        Notes
        -----
        - The method preserves the full spectral axis during computation.
        - If `update` is `False` and the model cube already exists, it returns the cached version.
        - For single-core computation, the `lazy_pix_compute_single` function is used.
        - For multi-core computation, the `custom_task_graph` function is used to process the model cube efficiently.
        - Chunk sizes are dynamically calculated using `calculate_chunks` for optimal memory usage.
        - The `calculate_batch_size` function determines the batch size for multi-core processing.

        Examples
        --------
        Generate the model cube using multi-core processing:

        >>> cube = MySpectralCube(...)  # A user-defined subclass of SpectralCube
        >>> model_cube = cube.get_modelcube(update=True, multicore=4, scheduler='threads')
        >>> print(model_cube.shape)
        (100, 200, 200)  # Example dimensions

        Use a mask to compute the model cube for specific regions:

        >>> mask = np.random.choice([True, False], size=(200, 200))
        >>> model_cube = cube.get_modelcube(mask=mask, multicore=False)
        """
        if self._modelcube is not None and not update:
            return self._modelcube

        if multicore is False:
            multicore = 1
        elif isinstance(multicore, int) and multicore > 0:
            multicore = True

        # update the has fit just in case there were changes
        isfinite = np.all(np.isfinite(self.parcube), axis=0)
        isvalid = isfinite & self._has_fit(get=True)

        if mask is not None:
            isvalid &= mask

        n_pix = np.sum(isvalid)

        if n_pix < 1:
            logger.warning("No valid pixels to update the model.")
            return self._modelcube

        if self._modelcube is None:
            # Calculate chunks
            chunks = dask_utils.calculate_chunks(cube_shape=self.cube.shape, dtype=self.cube.dtype,
                                                 target_chunk_mem_mb=target_memory_mb) #128

            # initializing modelcube
            self._modelcube = da.full(shape=self.cube.shape, fill_value=np.nan, dtype=self.cube.dtype,
                                chunks=chunks)

        def compute_pixel(x, y):
            """Compute the model spectrum for a single pixel."""
            return self.specfit.get_full_model(pars=self.parcube[:, y, x])

        if not multicore:
            # Sequential computation
            logger.debug("Running in single-core mode.")
            self._modelcube = dask_utils.lazy_pix_compute_single(self._modelcube, isvalid, compute_pixel)

        else:
            logger.debug("Running in threaded mode.")
            #self._modelcube = dask_utils.lazy_pix_compute_no_batching(self._modelcube, isvalid, compute_pixel)
            self._modelcube = dask_utils.custom_task_graph(self._modelcube, isvalid, compute_pixel,
                                                           use_global_xy=True, scheduler=scheduler)

        return self._modelcube

    def _has_fit(self, get=False):
        """
        Check if valid fits exist for each pixel in the parameter cube.

        Valid fits are determined by non-zero and finite parameter values.

        Parameters
        ----------
        get : bool, default=False
            If True, return the computed mask. If False, update `self.has_fit`.

        Returns
        -------
        numpy.ndarray, optional
            A 2D Boolean mask indicating valid fits for each pixel. Returned only
            if `get=True`.

        Notes
        -----
        - All-zero parameter slices are ignored in determining valid fits.
        - Updates `self.has_fit` when `get=False`.
        """
        if np.any(np.all(self.parcube == 0, axis=(1, 2))):
            # there are some slices where all parameters are zero, we should
            # ignore this when establishing whether there's a fit (some
            # parameters, like fortho, can be locked to zero)
            self.has_fit = np.all((np.isfinite(self.parcube)), axis=0)
        else:
            self.has_fit = np.all((self.parcube != 0) &
                                  (np.isfinite(self.parcube)), axis=0)

        if get:
            return self.has_fit

    def replace_para_n_mod(self, pcube_ref, planemask, multicore=None):
        """
        Replace parameter values in a parameter cube and update the model.

        Parameters
        ----------
        pcube_ref : pyspeckit.cube
            Reference parameter cube containing the values to replace in the target cube.
        planemask : numpy.ndarray
            A 2D Boolean mask where `True` indicates pixels to update.
        multicore : int or None, optional
            Number of cores to use for parallel computation. If `None`, a single core is used.

        Returns
        -------
        None

        Notes
        -----
        - Updates the following attributes of the target `PCube`:
          - `parcube`: Fitted parameter values.
          - `errcube`: Parameter errors.
          - `has_fit`: Boolean map indicating successful fits.
        - The `_modelcube` is updated using the corresponding model values from `pcube_ref`.
        - If `multicore` is provided, it controls the level of parallelism when updating the model cube.
        """
        # Replace values in masked pixels with the reference values
        self.parcube[:, planemask] = deepcopy(pcube_ref.parcube[:, planemask])
        self.errcube[:, planemask] = deepcopy(pcube_ref.errcube[:, planemask])

        # Update has_fit
        if isinstance(self, PCube):
            self._has_fit(get=False)
        else:
            self.has_fit[planemask] = deepcopy(pcube_ref.has_fit[planemask])

        # Validate the number of cores and update the model cube
        multicore = validate_n_cores(multicore)
        _ = self.get_modelcube(update=True, multicore=multicore, mask=planemask)