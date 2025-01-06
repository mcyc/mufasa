import numpy as np
import dask.array as da
from pyspeckit.cubes.SpectralCube import Cube

from .utils import dask_utils

#======================================================================================================================#
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#

class PCube(Cube):
    """
    A customized version of the SpectralCube class for Mufasa-specific
    enhancements, including memory-efficient model cube generation.
    """

    def get_modelcube(self, update=False, multicore=1, mask=None, target_memory_mb=None):
        """
        Customized model cube generation using Dask for parallel processing
        and memory efficiency, with masking support.

        Parameters:
        ----------
        update : bool
            Whether to force recalculation of the model cube.
        multicore : int
            Number of cores to use for parallel processing.
        mask : np.ndarray or None
            Boolean mask with the same shape as the cube's spatial dimensions
            (ny, nx). True indicates pixels to recalculate.
        target_memory_mb : int or None
            Desired memory usage per chunk in megabytes. If None, it is calculated
            based on available system memory.

        Returns:
        -------
        dask.array.Array
            The computed model cube as a Dask array.
        """
        if self._modelcube is not None and not update:
            return self._modelcube

        yy, xx = np.indices(self.parcube.shape[1:])
        nanvals = ~np.all(np.isfinite(self.parcube), axis=0)
        isvalid = np.any(self.parcube, axis=0) & ~nanvals

        if mask is not None:
            isvalid &= mask

        valid_pixels = list(zip(xx[isvalid], yy[isvalid]))

        if len(valid_pixels) < 1:
            logger.warning("No valid pixels to update the model.")
            return self._modelcube

        if self._modelcube is None:
            self._modelcube = np.full(self.cube.shape, np.nan, dtype=self.cube.dtype)

        def compute_pixel(x, y):
            """Compute the model spectrum for a single pixel."""
            return self.specfit.get_full_model(pars=self.parcube[:, y, x])

        if multicore == 1:
            # Sequential computation
            logger.deg("Running in single-core mode.")
            for x, y in valid_pixels:
                self._modelcube[:, y, x] = compute_pixel(x, y)

        else:
            # Parallel computation with Dask
            logger.info("Running in multi-core mode with Dask.")
            self._modelcube = dask_utils.parallel_compute(
                host_cube=self._modelcube,
                valid_pixels=valid_pixels,
                compute_pixel=compute_pixel,
                multicore=multicore,
                target_memory_mb=target_memory_mb,
                max_usable_fraction=0.85,
                logger=logger
            )

        return self._modelcube
