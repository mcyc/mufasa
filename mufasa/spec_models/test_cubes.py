"""
Provides functionality to generate mock spectral cubes using mufasa's spectral models.
"""

import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.stats import skewnorm, norm
import random
from astropy.convolution import convolve_fft, Gaussian2DKernel


class MockCloud(object):
    """
    Generate synthetic 2D parameter maps for a single molecular cloud component.

    Fields are generated from an underlying 2D random field with a power-law
    power spectrum (see `generate_powerlaw_field_pixel_based`). Passing the
    same seed to multiple `get_*` methods yields spatially correlated maps,
    mimicking the physical relationships expected in a real molecular cloud
    (e.g., regions of high column density tending to have narrower linewidths).

    Parameters
    ----------
    box_size : int, optional
        Size of the grid in pixels (e.g., 256 for a 256x256 map). Default is 256.
    pixel_size : float, optional
        Physical size of each pixel, in parsecs. Default is 0.01.

    Attributes
    ----------
    box_size : int
        Grid size in pixels, as passed at construction.
    pixel_size : float
        Pixel scale in parsecs, as passed at construction.
    largest_scale : float
        Physical size of the full map (box_size * pixel_size), in parsecs.
    beta : float
        Power-law index of the column density power spectrum.
    lognorm_kw : dict
        Parameters (mean, std) for the column density log-normal distribution.
    vlos_kw : dict
        Parameters (alpha, beta, coherent_scale, std) for the line-of-sight
        velocity field.
    sigv_kw : dict
        Parameters (field_sign, mean_log, std_log) for the velocity dispersion
        log-normal distribution.
    field, field_log_normal, field_vlos, field_sigv : ndarray or None
        Cached fields from the most recent `get_*` call of the corresponding
        type. None until the relevant method has been called at least once.
    """

    def __init__(self, box_size=256, pixel_size=0.01):

        self.seed = None
        self.seed2 = None
        self.set_seed(42)  # set default seed 1 & 2

        self.box_size = box_size
        self.pixel_size = pixel_size
        self.largest_scale = box_size * pixel_size  # parsecs

        self.beta = 2.7  # spectral index of column density power spectrum

        self.lognorm_kw = dict(
            mean=1.0,  # Mean column density (arbitrary units)
            std=0.3  # Standard deviation of the log-normal distribution
        )

        # Note - for VLOS's beta:
        # empirical beta ~2.8-3.2 (Elmegreen & Scalo 2004)
        # simulation beta ~ 2.7 (Padoan et al. 2003)

        self.vlos_kw = dict(
            alpha=0.5,  # Larson's relation index
            beta=2 * 0.5 + 2,  # Comforms to Larson's relation and Burgers' turbulence
            coherent_scale=0.5,  # pc, the scale for which velocity has roughly the same structure as the column density
            std=0.75  # km/s, the standard deviation to normalize the velocity field standard deviation
        )

        # these default values mimic GAS & KEYSTONE NH3 results
        self.sigv_kw = dict(
            field_sign=-1, # (-1 or 1) Default of -1 means linewidth anti-correlates with column density
            mean_log=0.45,  # Mean of the log-normal (arbitrary units)
            std_log=0.6  # Standard deviation of the log-normal distribution
        )

        self.field = None
        self.field_log_normal = None
        self.field_vlos = None
        self.field_sigv = None

    def set_seed(self, seed):
        """
        Set the primary seed and derive a secondary seed from it.

        Parameters
        ----------
        seed : int or None
            Seed for `random` and downstream field generation. If None,
            no-op (current seed and seed2 are left unchanged).
        """
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            self.seed2 = random.randint(1, 42000)  # pick a random integer in that range

    def isnewseed(self, seed):
        """
        Check whether `seed` differs from the currently stored seed.

        Parameters
        ----------
        seed : int or None
            Seed to compare against `self.seed`. None always returns False,
            signaling "no new seed requested, reuse cached field if available."

        Returns
        -------
        new : bool
            True if `seed` is not None and differs from `self.seed`.
        """

        if seed is None:
            new = False
        else:
            new = seed != self.seed

        return new

    def get_powerlaw_field(self, seed=None):
        """
        Generate (or retrieve the cached) 2D power-law random field.

        This is the base field from which `get_lognormal_field` derives the
        column density proxy map. Calling with the same seed as the last call
        returns the cached field rather than regenerating it.

        Parameters
        ----------
        seed : int, optional
            Random seed for field generation. If None, reuses the most
            recently generated field (or the default seed if none exists yet).

        Returns
        -------
        field : ndarray
            2D power-law random field, normalized to zero mean and unit variance.
        """

        new = self.isnewseed(seed)

        if new or self.field is None:
            self.set_seed(seed)
            # generate a new powerlaw field
            kwargs = dict(
                box_size=self.box_size,
                pixel_size=self.pixel_size,
                beta=self.beta,
                random_seed=self.seed
            )
            self.field = generate_powerlaw_field_pixel_based(**kwargs)

        return self.field

    def get_lognormal_field(self, seed=None, invert=False):
        """
        Generate a log-normal column density proxy map.

        Built by exponentiating a scaled version of the underlying power-law
        field (see `get_powerlaw_field`), consistent with the column density
        statistics of isothermal, supersonic turbulence (e.g., Vazquez-Semadeni
        1994; Padoan et al. 2003).

        Parameters
        ----------
        seed : int, optional
            Random seed for field generation. If None, reuses the most
            recently generated field (or the default seed if none exists yet).
        invert : bool, optional
            If True, flip the sign of the underlying power-law field before
            exponentiating, reversing which spatial structures correspond to
            density peaks vs. troughs. Default is False.

        Returns
        -------
        field_log_normal : ndarray
            2D log-normal column density proxy map, with mean and standard
            deviation set by `self.lognorm_kw`.
        """
        # Generate the power-law field

        if self.isnewseed(seed) or self.field_log_normal is None:

            field = self.get_powerlaw_field(seed)
            if invert:
                field *= -1
            mean = self.lognorm_kw['mean']
            std = self.lognorm_kw['std']

            # Scale and exponentiate to create log-normal distribution
            field_scaled = np.log(mean) - 0.5 * (std ** 2) + std * field
            self.field_log_normal = np.exp(field_scaled)

        return self.field_log_normal

    def get_kinetic_powerlaw_field(self, seed=None, seed2=None):
        """
        Generate a 2D power-law random field using the velocity power spectrum.

        This is the shared base field used by both `get_velocity_field` and
        `get_sigma_v`, so that the line-of-sight velocity and velocity
        dispersion maps share the same underlying turbulent structure
        (scaled by `self.vlos_kw['beta']` and `self.vlos_kw['coherent_scale']`).

        Parameters
        ----------
        seed : int, optional
            Random seed for field generation. If None, reuses `self.seed`.
        seed2 : int or True, optional
            Secondary seed used to modify the phase of large-scale structures
            (see `generate_powerlaw_field_pixel_based`). If True, uses
            `self.seed2`. If None, no large-scale phase modification is applied.

        Returns
        -------
        field : ndarray
            2D power-law random field, normalized to zero mean and unit variance.
        """
        # TODO: the seed/seed2 interaction here needs further investigation —
        # self.seed2 is read before self.set_seed(seed) updates self.seed below,
        # so seed2=True may reflect a prior call's secondary seed rather than
        # one freshly tied to the current `seed`. Likely to be superseded by
        # the upcoming MockCloud/MockComponent restructuring.
        if seed2 is True:
            seed2 = self.seed2

        if seed is not None:
            self.set_seed(seed)

        kwargs = dict(
            box_size=self.box_size,
            pixel_size=self.pixel_size,
            beta=self.vlos_kw['beta'],  # specific for velocity
            random_seed=self.seed,
            random_seed_2=seed2,
            length_scale=self.vlos_kw['coherent_scale'] / self.pixel_size,  # coherent_scale in pixel unit
        )
        return generate_powerlaw_field_pixel_based(**kwargs)


    def get_velocity_field(self, seed=None, seed2=True, skewness=None):

        field_vlos = self.get_kinetic_powerlaw_field(seed=seed, seed2=seed2)

        def standardize(field_vlos):
            # ensure normalization
            field_vlos -= np.mean(field_vlos)
            field_vlos /= np.std(field_vlos)
            return field_vlos

        field_vlos = standardize(field_vlos)

        if skewness is not None and skewness != 0:
            # ensure normalization
            # transform into a skewed Gaussian
            a = -skewness
            norm_cdf = norm.cdf(field_vlos)
            field_vlos = skewnorm.ppf(norm_cdf, a)
            # recenter
            field_vlos = standardize(field_vlos)

        # re-normalize the standard deviation of the vlos distribution
        scaling_factor = self.vlos_kw['std'] / np.std(field_vlos)
        self.field_vlos = field_vlos * scaling_factor

        if self.field_vlos is not None:
            return self.field_vlos


    def get_sigma_v(self, seed=None):
        field_pl = self.get_kinetic_powerlaw_field(seed=seed)
        field_pl *= self.sigv_kw['field_sign']  # correlate or anti-correlate with column density

        # Scale and exponentiate to create log-normal distribution for sigma_v
        mean = self.sigv_kw['mean_log']
        std = self.sigv_kw['std_log']
        field_scaled = np.log(mean) - 0.5 * (std ** 2) + std * field_pl
        self.field_sigv = np.exp(field_scaled)
        return self.field_sigv


    def get_column_density(self):
        pass

    def get_tau(self):
        # scales with log-normal power spectrum, normalized tp 0.1 - 8?
        pass

    def get_tex(self):
        # uniform box function like distribution in [4-8] K?
        pass


def generate_powerlaw_field_pixel_based(box_size, pixel_size, beta, random_seed=None, random_seed_2=None,
                                        length_scale=None):
    """
    Generate a 2D random field with a power-law power spectrum. Optionally, modify the phase of larger scale structures.

    Parameters
    ----------
    box_size : int
        Size of the grid in pixels (e.g., 512 for a 512x512 image).
    pixel_size : float
        Physical size of each pixel (e.g., in parsecs).
    beta : float
        Power-law index of the spectrum (e.g., 2.7 for turbulence).
    random_seed : int, optional
        Random seed for reproducibility.
    random_seed_2 : int, optional
        Second random seed to modify the phase of large-scale structures.
    length_scale : float, optional
        Length scale (in pixel units) to separate small and large structures. If None, no phase modification occurs.

    Returns
    -------
    field : ndarray
        2D array with the generated random field.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Total physical size of the grid
    physical_size = box_size * pixel_size

    # Grid of frequencies (in units of 1/physical_size)
    kx = np.fft.fftfreq(box_size, d=pixel_size)
    ky = np.fft.fftfreq(box_size, d=pixel_size)
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k_squared = kx ** 2 + ky ** 2

    # Apply a small regularization to avoid division by zero at k = 0
    k_squared[k_squared == 0] = 1e-10

    # Power-law spectrum
    power_spectrum = (k_squared) ** (-beta / 2)

    # Generate random noise in Fourier space
    random_noise = np.random.normal(size=(box_size, box_size)) + 1j * np.random.normal(size=(box_size, box_size))

    # Apply the power spectrum
    correlated_noise = fftn(random_noise) * np.sqrt(power_spectrum)

    # Optionally modify the phase of large-scale structures
    if random_seed_2 is not None and length_scale is not None:
        # Length scale in frequency space
        k_threshold = 1.0 / (length_scale * pixel_size)

        # Mask for large scales (k < k_threshold)
        large_scale_mask = np.sqrt(k_squared) < k_threshold

        # Generate new random noise for large scales
        np.random.seed(random_seed_2)

        rn2 = np.random.normal(size=(box_size, box_size)) + 1j * np.random.normal(size=(box_size, box_size))

        correlated_noise[large_scale_mask] = fftn(rn2)[large_scale_mask] * (power_spectrum ** 0.25)[large_scale_mask]

    # Transform back to real space
    field = np.real(ifftn(correlated_noise))

    # Normalize to zero mean and unit variance
    field -= np.mean(field)
    field /= np.std(field)

    return field


# ======================================================================================================================================

def generate_powerlaw_field_3d(box_size, pixel_size, beta, random_seed=None):
    """
    Generate a 3D random field with a power-law power spectrum.

    Parameters
    ----------
    box_size : int
        Size of the grid in pixels along one dimension (e.g., 128 for a 128x128x128 cube).
    pixel_size : float
        Physical size of each pixel (e.g., in parsecs).
    beta : float
        Power-law index of the spectrum
    random_seed : int, optional
        Random seed for reproducibility.

    Notes
    -------
    - beta (velocity) beta = 2+2  # Kritsuk et al. (2010) and Federrath et al. (2010), 1.76 in subsonic regime Federrath et al. (2021)
    - beta (density) beta= 3.3 #Burkhart et al. (2015), Kritsuk et al. (2010), Auddy et al. (2017), beta=5: (Burgers trubulence)

    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Total physical size of the grid
    physical_size = box_size * pixel_size

    # Grid of frequencies (in units of 1/physical_size)
    kx = np.fft.fftfreq(box_size, d=pixel_size)
    ky = np.fft.fftfreq(box_size, d=pixel_size)
    kz = np.fft.fftfreq(box_size, d=pixel_size)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    k_squared = kx ** 2 + ky ** 2 + kz ** 2

    # Avoid division by zero at k = 0
    k_squared[k_squared == 0] = 1e-10

    # Power-law spectrum
    power_spectrum = (k_squared) ** (-beta / 2)

    # Generate random noise in Fourier space
    random_noise = (np.random.normal(size=(box_size, box_size, box_size)) +
                    1j * np.random.normal(size=(box_size, box_size, box_size)))

    # Apply the power spectrum
    correlated_noise = fftn(random_noise) * np.sqrt(power_spectrum)

    # Transform back to real space
    field = np.real(ifftn(correlated_noise))

    # Normalize to zero mean and unit variance
    field -= np.mean(field)
    field /= np.std(field)

    return field


def normalize_to_lognormal_density_field(density_field, mean_density=1.0, density_std=1.0):
    """
    Normalize a 3D density field to follow a lognormal distribution.

    Parameters
    ----------
    density_field : ndarray
        Input 3D array representing the initial density field (can have negative values).
    mean_density : float, optional
        Mean density of the output field in physical units. Default is 1.0.
    density_std : float, optional
        Standard deviation of the density field in log-space. Default is 1.0.

    Returns
    -------
    lognormal_density_field : ndarray
        3D array representing the lognormal density field.
    """
    # Normalize the input field to have zero mean and unit variance
    density_field -= np.mean(density_field)
    density_field /= np.std(density_field)

    # Convert the Gaussian field to a lognormal field
    lognormal_field = np.exp(density_std * density_field)

    # Rescale the lognormal field to have the desired mean density
    lognormal_field *= mean_density / np.mean(lognormal_field)

    return lognormal_field


def density_weighted_mean_and_std_axis(quantity, density, axis=0):
    """
    Calculate the density-weighted mean and standard deviation along a specified axis.

    Parameters
    ----------
    quantity : ndarray
        Array of the quantity values (e.g., temperature, velocity).
    density : ndarray
        Array of density values corresponding to the quantity.
    axis : int, optional
        Axis along which to calculate the weighted statistics. Default is 0.

    Returns
    -------
    weighted_mean : ndarray
        Density-weighted mean along the specified axis.
    weighted_std : ndarray
        Density-weighted standard deviation along the specified axis.
    """
    # Calculate the weighted mean along the specified axis
    weighted_mean = np.sum(density * quantity, axis=axis) / np.sum(density, axis=axis)

    # Calculate the weighted standard deviation along the specified axis
    weighted_std = np.sqrt(
        np.sum(density * (quantity - np.expand_dims(weighted_mean, axis=axis)) ** 2, axis=axis)
        / np.sum(density, axis=axis)
    )

    return weighted_mean, weighted_std