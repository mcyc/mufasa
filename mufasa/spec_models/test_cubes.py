"""
Provides functionality to generate mock spectral cubes using mufasa's spectral models.
"""

import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from astropy.convolution import convolve_fft, Gaussian2DKernel


class MockCloud(object):

    def __init__(self, box_size=256, pixel_size=0.01):

        self.seed = 42 #default

        self.box_size = box_size
        self.pixel_size = pixel_size
        self.largest_scale = box_size * pixel_size  # parsecs

        self.beta = 2.7  # spectral index of column density power spectrum

        self.lognorm_kw = dict(
            mean=1.0,  # Mean column density (arbitrary units)
            std=0.3  # Standard deviation of the log-normal distribution
        )

        self.vlos_kw = dict(
            alpha=0.5,  # Larson's relation index
            velocity_scale=0.75,  # km/s at the largest scale
        )

        self.field=None
        self.field_log_normal=None
        self.vlos_field=None

    def set_seed(self, seed):
        if seed is not None:
            self.seed = seed

    def isnewseed(self, seed):

        if seed is None:
            new = False
        else:
            new = seed != self.seed

        return new


    def get_powerlaw_field(self, seed=None):

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

    def get_velocity_field(self, seed=None):

        print(f"{seed} is a new seed: {self.isnewseed(seed)}")

        if self.isnewseed(seed) or self.vlos_field is None:

            field = self.get_powerlaw_field(seed)

            # Standard deviation of the generated field
            field_std = np.std(field)
            velocity_scale = self.vlos_kw['velocity_scale']

            # Calculate the expected velocity dispersion at the largest scale
            expected_sigma_v = velocity_scale

            # Scaling factor based on Larson's relation
            scaling_factor = expected_sigma_v / field_std

            # Scale the field
            self.velocity_field = field * scaling_factor

            # Optionally verify the scaling follows Larson's relation
            # Compute the physical scale of the simulation box
            simulation_box_size = field.shape[0] * self.pixel_size

            # Check the velocity dispersion at intermediate scales if needed (not implemented here)

        return self.velocity_field

    def get_sigma_v(self):
        pass

    def get_column_density(self):
        pass

    def get_tau(self):
        pass

    def get_tex(self):
        pass



def generate_powerlaw_field_pixel_based(box_size, pixel_size, beta, random_seed=None):
    """
    Generate a 2D random field with a power-law power spectrum.

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

    # Transform back to real space
    field = np.real(ifftn(correlated_noise))

    # Normalize to zero mean and unit variance
    field -= np.mean(field)
    field /= np.std(field)

    return field


#======================================================================================================================================

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
    k_squared = kx**2 + ky**2 + kz**2

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
        np.sum(density * (quantity - np.expand_dims(weighted_mean, axis=axis))**2, axis=axis)
        / np.sum(density, axis=axis)
    )

    return weighted_mean, weighted_std

