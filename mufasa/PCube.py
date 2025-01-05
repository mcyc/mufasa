import numpy as np
import dask.array as da
from dask import delayed
from dask.distributed import Client
from pyspeckit.cubes.SpectralCube import Cube
import psutil  # To detect system memory

#======================================================================================================================#
from .utils.mufasa_log import get_logger
logger = get_logger(__name__)


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

        model_cube_shape = self.cube.shape
        self._modelcube = np.full(model_cube_shape, np.nan, dtype=self.cube.dtype)

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

            def compute_chunk(pixel_chunk):
                """Compute the model spectra for a chunk of pixels."""
                chunk_data = np.empty((model_cube_shape[0], len(pixel_chunk)), dtype=self.cube.dtype)
                for i, (x, y) in enumerate(pixel_chunk):
                    chunk_data[:, i] = compute_pixel(x, y)
                return chunk_data

            if target_memory_mb is None:
                total_memory = psutil.virtual_memory().total  # Total system memory in bytes
                target_memory_mb = total_memory * 0.05 / (1024 * 1024)  # Use 5% of total memory

            memory_per_spectrum = model_cube_shape[0] * np.dtype(self.cube.dtype).itemsize
            chunk_size = max(1, int((target_memory_mb * 1024 * 1024) / memory_per_spectrum))
            chunk_size = min(chunk_size, len(valid_pixels))

            pixel_chunks = [valid_pixels[i:i + chunk_size] for i in range(0, len(valid_pixels), chunk_size)]
            logger.debug(f"Chunk size: {chunk_size}")
            logger.debug(f"Number of pixel chunks: {len(pixel_chunks)}")

            client = Client(n_workers=multicore) if multicore > 1 else None

            results = []
            for chunk in pixel_chunks:
                result = da.from_delayed(
                    delayed(compute_chunk)(chunk),
                    shape=(model_cube_shape[0], len(chunk)),
                    dtype=self.cube.dtype
                )
                results.append(result)

            if results:
                all_data = da.concatenate(results, axis=1)
                index = 0
                for chunk in pixel_chunks:
                    for x, y in chunk:
                        self._modelcube[:, y, x] = all_data[:, index]
                        index += 1

            if client:
                client.close()

        return self._modelcube
