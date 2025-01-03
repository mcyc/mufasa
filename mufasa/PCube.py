import numpy as np
import dask.array as da
from dask import delayed
from pyspeckit.cubes.SpectralCube import Cube


class PCube(Cube):
    """
    A customized version of the SpectralCube class for Mufasa-specific
    enhancements, including memory-efficient model cube generation.
    """

    def get_modelcube(self, update=False, multicore=1, mask=None):
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

        Returns:
        -------
        dask.array.Array
            The computed model cube as a Dask array.
        """
        if self._modelcube is not None and not update:
            return self._modelcube

        yy, xx = np.indices(self.parcube.shape[1:])
        nanvals = np.any(~np.isfinite(self.parcube), axis=0)
        isvalid = np.any(self.parcube, axis=0) & ~nanvals

        if mask is not None:
            isvalid &= mask

        valid_pixels = list(zip(xx[isvalid], yy[isvalid]))

        # Create a lazy Dask array to hold the model cube
        model_cube_shape = self.cube.shape
        self._modelcube = da.full(model_cube_shape, np.nan, dtype=self.cube.dtype)

        def compute_model(x, y):
            """Compute the model spectrum for a single pixel."""
            return self.specfit.get_full_model(pars=self.parcube[:, y, x])

        # Generate the Dask graph for valid pixels
        for x, y in valid_pixels:
            self._modelcube[:, y, x] = da.from_delayed(
                delayed(compute_model)(x, y),
                shape=(model_cube_shape[0],),  # Shape of the spectral axis
                dtype=self.cube.dtype
            )

        return self._modelcube
