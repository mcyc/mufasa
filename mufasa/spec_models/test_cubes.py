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
    Orchestrate a collection of `MockComponent` instances sharing a common grid.

    `MockCloud` owns the spatial grid (`box_size`, `pixel_size`) and holds a
    list of `MockComponent` instances, one per cloud component, each
    representing an independent set of column density, velocity dispersion,
    and line-of-sight velocity maps. The only thing that couples components
    together is their relative placement along the velocity axis, applied at
    construction via `apply_velocity_offsets`.

    Parameters
    ----------
    box_size : int, optional
        Size of the grid in pixels (e.g., 256 for a 256x256 map). Default is 256.
    pixel_size : float, optional
        Physical size of each pixel, in parsecs. Default is 0.01.
    n_components : int, optional
        Number of components to create at construction. Default is 1.
    seeds : list of int, optional
        Seed for each component, in order. If None, seeds are auto-generated
        as 42, 43, 44, ... (one per component) to ensure components are
        statistically independent by default. If provided, must have length
        `n_components`.
    v_offsets : float or list of float, optional
        Spacing between consecutive components' v_los fields, passed to
        `apply_velocity_offsets` at the end of construction. If None
        (default), components are spaced by 1.0 (one velocity dispersion,
        i.e. each component's own `vlos_kw['std']`). See
        `apply_velocity_offsets` for the full placement convention.

    Attributes
    ----------
    box_size : int
        Grid size in pixels, as passed at construction.
    pixel_size : float
        Pixel scale in parsecs, as passed at construction.
    largest_scale : float
        Physical size of the full map (box_size * pixel_size), in parsecs.
    components : list of MockComponent
        The cloud's components, in order. Each component's `field_vlos` has
        already been placed (shifted/sign-flipped) by `apply_velocity_offsets`
        by the time `__init__` returns.
    """

    def __init__(self, box_size=256, largest_scale=2.5, n_components=1, seeds=None, v_offsets=None,
                 vlos_std=0.5, coherent_scale=0.5, column_density_pdf="lognormal"):

        self.box_size = box_size
        #self.pixel_size = pixel_size
        #self.largest_scale = box_size * pixel_size  # parsecs
        self.largest_scale = largest_scale # parsecs
        self.pixel_size = largest_scale/box_size
        self.column_density_pdf = column_density_pdf


        if seeds is None:
            seeds = [42 + i for i in range(n_components)]
        elif len(seeds) != n_components:
            raise ValueError(
                f"len(seeds)={len(seeds)} does not match n_components={n_components}"
            )

        self.components = [
            MockComponent(box_size=self.box_size, pixel_size=self.pixel_size, seed=seed, vlos_std=vlos_std, coherent_scale=coherent_scale, column_density_pdf=column_density_pdf)
            for seed in seeds
        ]

        # place components' v_los fields along the velocity axis; defaults
        # to 1 velocity dispersion of spacing if v_offsets is not given
        # self.apply_velocity_offsets(1.0 if v_offsets is None else v_offsets)
        if v_offsets:
            self.apply_velocity_offsets(v_offsets)

    def add_component(self, seed=None):
        """
        Append a new component to `self.components`.

        Note this does not re-run `apply_velocity_offsets` — the newly added
        component's `field_vlos` remains unplaced (raw) until
        `apply_velocity_offsets` is called again explicitly across the full,
        updated `self.components` list.

        Parameters
        ----------
        seed : int, optional
            Seed for the new component. If None, uses `MockComponent`'s
            default seed (42) — note this may duplicate an existing
            component's seed if not set explicitly.

        Returns
        -------
        component : MockComponent
            The newly created and appended component.
        """
        kwargs = dict(
            box_size=self.box_size,
            pixel_size=self.pixel_size,
            column_density_pdf=self.column_density_pdf,
        )
        if seed is not None:
            kwargs['seed'] = seed
        component = MockComponent(**kwargs)
        self.components.append(component)
        return component

    def apply_velocity_offsets(self, offsets):
        """
        Place each component's v_los field along the velocity axis, in
        ascending order, centered/sign-flipped per a fixed convention.

        Unlike most `get_*` methods in this module, this method does not
        generate new fields — it reads each component's *current*
        `field_vlos` (the raw, unplaced field cached by `MockComponent` at
        construction) and overwrites it in place with the shifted/flipped
        result.

        .. warning::
            Calling this method more than once **compounds** the placement,
            since each call places whatever is currently in `field_vlos`,
            not the original raw field. Call it only once per desired
            placement (as `MockCloud.__init__` does automatically); to
            re-place from scratch, construct a new `MockCloud`, or manually
            regenerate each component's raw field first via
            `component.field_vlos = component.get_velocity_field(component.seed)`
            (which always regenerates fresh, since `get_velocity_field` does
            not cache/early-return) before calling this again.

        Components are placed in the order given by `self.components`, from
        lowest to highest velocity centroid, spaced according to `offsets`.
        Positions are then shifted so that the midpoint between the lowest
        and highest position sits at 0 (symmetric placement for any `n`;
        for odd `n` with evenly-spaced gaps this also puts the middle
        component exactly at 0). Sign flipping follows:

        - 1 component: never sign-flipped.
        - 2 components: the higher (second) component has its v_los field
          sign-flipped, mirroring its turbulent structure in addition to
          shifting it. This matches the common observational scenario of
          two overlapping but kinematically distinct cloud components.
        - >2 components: each component independently has a ~50/50 chance
          of being sign-flipped, drawn deterministically from that
          component's own seed (so the flip pattern is reproducible for a
          given set of component seeds).

        Parameters
        ----------
        offsets : float or list of float
            Spacing between consecutive components, in units of each
            component's own `vlos_kw['std']` (i.e. as a multiple of its
            velocity dispersion). If a scalar, components are evenly spaced
            by that amount. If a list, must have length
            `len(self.components) - 1`, giving the gap between each
            consecutive pair of components.

        Returns
        -------
        fields : list of ndarray
            Placed v_los map for each component, in order (same arrays as
            `[c.field_vlos for c in self.components]` after this call).
        """
        n = len(self.components)

        if n == 1:
            gaps = []
        elif np.isscalar(offsets):
            gaps = [offsets] * (n - 1)
        else:
            if len(offsets) != n - 1:
                raise ValueError(
                    f"len(offsets)={len(offsets)} does not match "
                    f"len(self.components) - 1 = {n - 1}"
                )
            gaps = list(offsets)

        # cumulative positions, ascending, before centering
        raw_positions = np.concatenate([[0.0], np.cumsum(gaps)])

        # center on the midpoint between the lowest and highest position,
        # so placement is symmetric regardless of n's parity
        midpoint = (raw_positions[0] + raw_positions[-1]) / 2
        positions = raw_positions - midpoint

        # determine sign flips
        flips = np.ones(n)
        if n == 2:
            flips[1] = -1
        elif n > 2:
            for i, component in enumerate(self.components):
                rng = np.random.RandomState(component.seed)
                if rng.rand() < 0.5:
                    flips[i] = -1

        fields = []
        for component, position, flip_sign in zip(self.components, positions, flips):
            sigv = component.vlos_kw['std']
            placed = flip_sign * component.field_vlos + sigv * position
            component.field_vlos = placed
            fields.append(placed)

        return fields


class MockComponent(object):
    """
    Generate synthetic 2D parameter maps for a single molecular cloud component.

    This is the per-component counterpart to `MockCloud`: it owns the same
    field-generation logic, but takes the spatial grid (`box_size`,
    `pixel_size`) as constructor arguments rather than owning them itself, so
    that multiple components can share a common grid when assembled into a
    multi-component synthetic cloud (see `MockCloud`).

    Parameters
    ----------
    box_size : int
        Size of the grid in pixels (e.g., 256 for a 256x256 map).
    pixel_size : float
        Physical size of each pixel, in parsecs.
    seed : int, optional
        Initial seed for this component's field generation. Default is 42.

    Attributes
    ----------
    box_size : int
        Grid size in pixels, as passed at construction.
    pixel_size : float
        Pixel scale in parsecs, as passed at construction.
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
    field : ndarray
        Underlying power-law random field used to derive `field_column_density`.
    field_column_density : ndarray
        Column density proxy map, generated at construction via
        `get_lognormal_field` using `seed`. Named for the physical quantity
        rather than the generating method, since column density may be
        derived through other means (and may require unit scaling) in the
        future.
    field_vlos : ndarray
        Line-of-sight velocity map, generated at construction via
        `get_velocity_field` using `seed`.
    field_sigv : ndarray
        Velocity dispersion map, generated at construction via `get_sigma_v`
        using `seed`.

    Notes
    -----
    All four field attributes above are populated once at construction time
    using `seed`, and are then cached/overwritten in place if the
    corresponding `get_*` method is called again with a different seed.
    """

    def __init__(self, box_size, pixel_size, seed=42, vlos_std=0.5, coherent_scale=0.5, column_density_pdf="lognormal"):

        self.seed = None
        self.seed2 = None
        self.set_seed(seed)  # set default seed 1 & 2

        self.box_size = box_size
        self.pixel_size = pixel_size

        self.beta = 2.7  # spectral index of column density power spectrum

        self.lognorm_kw = dict(
            mean=1.0,  # Mean column density (arbitrary units)
            std=0.3  # Standard deviation of the log-normal distribution
        )
        self.powerlaw_pdf_kw = dict(
            alpha=2.0,  # Differential PDF slope, p(N) ∝ N**(-alpha); default based on Sadavoy+ 2014's overall Perseus PDF
            xmin=0.1,
            mean=1.0,
        )
        self.column_density_pdf = None
        self._field_column_density_kind = None
        self.set_column_density_pdf(column_density_pdf)

        # Note - for VLOS's beta:
        # empirical beta ~2.8-3.2 (Elmegreen & Scalo 2004)
        # simulation beta ~ 2.7 (Padoan et al. 2003)

        self.vlos_kw = dict(
            # alpha=0.5,  # Larson's relation index
            beta=2 * 0.5 + 2,  # Comforms to Larson's relation and Burgers' turbulence
            coherent_scale=coherent_scale,  # pc, the scale for which velocity has roughly the same structure as the column density
            std=vlos_std  # km/s, the standard deviation to normalize the velocity field standard deviation
        )

        # these default values mimic GAS & KEYSTONE NH3 results
        self.sigv_kw = dict(
            field_sign=-1, # (-1 or 1) Default of -1 means linewidth anti-correlates with column density
            mean_log=0.45,  # Mean of the log-normal (arbitrary units)
            std_log=0.6  # Standard deviation of the log-normal distribution
        )

        self.field = None
        self.field_column_density = None
        self.field_vlos = None
        self.field_sigv = None

        # eagerly populate all fields at construction so they're immediately
        # available as attributes, without requiring an explicit get_* call
        self.field_column_density = self.get_column_density(self.seed)
        self.field_sigv = self.get_sigma_v(self.seed)
        self.field_vlos = self.get_velocity_field(self.seed)

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

    def set_column_density_pdf(self, column_density_pdf):
        """
        Set the one-point PDF used for the column density proxy map.
        """
        kind, value = self._parse_column_density_pdf(column_density_pdf)

        if kind == "lognormal" and value is not None:
            self.lognorm_kw["std"] = value
        elif kind == "powerlaw" and value is not None:
            self.powerlaw_pdf_kw["alpha"] = value

        self.column_density_pdf = kind
        self.field_column_density = None
        self._field_column_density_kind = None
        return kind

    @staticmethod
    def _parse_column_density_pdf(column_density_pdf):
        if isinstance(column_density_pdf, str):
            kind = column_density_pdf
            value = None
        elif (
            isinstance(column_density_pdf, tuple)
            and len(column_density_pdf) == 2
            and isinstance(column_density_pdf[0], str)
        ):
            kind, value = column_density_pdf
        else:
            raise TypeError(
                "column_density_pdf must be a string or a tuple of "
                "(name, value), e.g. 'lognormal' or ('powerlaw', 2.5)."
            )

        aliases = {
            "lognormal": "lognormal",
            "log-normal": "lognormal",
            "ln": "lognormal",
            "powerlaw": "powerlaw",
            "power-law": "powerlaw",
            "pl": "powerlaw",
        }
        key = kind.lower().replace("_", "-")
        if key not in aliases:
            raise ValueError(
                "column_density_pdf must be one of 'lognormal' or 'powerlaw'."
            )

        if value is not None and not np.isscalar(value):
            raise TypeError("The tuple value in column_density_pdf must be a scalar.")

        return aliases[key], value

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
        field_column_density : ndarray
            2D log-normal column density proxy map, with mean and standard
            deviation set by `self.lognorm_kw`.
        """
        # Generate the power-law field

        if (
            self.isnewseed(seed)
            or self.field_column_density is None
            or self._field_column_density_kind != "lognormal"
        ):

            field = self.get_powerlaw_field(seed).copy()
            if invert:
                field *= -1
            mean = self.lognorm_kw['mean']
            std = self.lognorm_kw['std']

            # Scale and exponentiate to create log-normal distribution
            field_scaled = np.log(mean) - 0.5 * (std ** 2) + std * field
            self.field_column_density = np.exp(field_scaled)
            self._field_column_density_kind = "lognormal"

        return self.field_column_density

    def get_powerlaw_column_density(self, seed=None, invert=False):
        """
        Generate a column density proxy map with a power-law one-point PDF.
        """
        if (
            self.isnewseed(seed)
            or self.field_column_density is None
            or self._field_column_density_kind != "powerlaw"
        ):
            field = self.get_powerlaw_field(seed).copy()
            if invert:
                field *= -1

            alpha = self.powerlaw_pdf_kw["alpha"]
            xmin = self.powerlaw_pdf_kw["xmin"]
            mean = self.powerlaw_pdf_kw["mean"]
            if alpha <= 1:
                raise ValueError("powerlaw_pdf_kw['alpha'] must be > 1.")
            if xmin <= 0:
                raise ValueError("powerlaw_pdf_kw['xmin'] must be > 0.")

            u = norm.cdf(field)
            eps = np.finfo(float).eps
            u = np.clip(u, eps, 1 - eps)

            field_column_density = xmin * (1 - u) ** (-1 / (alpha - 1))
            if mean is not None:
                field_column_density *= mean / np.mean(field_column_density)

            self.field_column_density = field_column_density
            self._field_column_density_kind = "powerlaw"

        return self.field_column_density

    def get_column_density(self, seed=None, pdf=None, invert=False):
        if pdf is not None:
            self.set_column_density_pdf(pdf)

        if self.column_density_pdf == "lognormal":
            return self.get_lognormal_field(seed=seed, invert=invert)
        if self.column_density_pdf == "powerlaw":
            return self.get_powerlaw_column_density(seed=seed, invert=invert)

        raise RuntimeError(f"Unknown column_density_pdf: {self.column_density_pdf!r}")

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
        """
        Generate a line-of-sight velocity (v_los) map.

        Built from the same power-law field family as `get_sigma_v` (see
        `get_kinetic_powerlaw_field`), standardized to zero mean and unit
        variance, then re-scaled to the standard deviation set by
        `self.vlos_kw['std']`. The power-law index follows Larson's relation
        and Burgers' turbulence scaling.

        Parameters
        ----------
        seed : int, optional
            Random seed for field generation. If None, reuses `self.seed`.
        seed2 : int or True, optional
            Secondary seed for large-scale phase modification, forwarded to
            `get_kinetic_powerlaw_field`. Default is True (use `self.seed2`).
        skewness : float, optional
            If provided and nonzero, transform the field into a skewed
            Gaussian with this skewness, then re-center and re-normalize.
            Useful for simulating non-Gaussian turbulent velocity
            distributions. Default is None (no skew).

        Returns
        -------
        field_vlos : ndarray
            2D line-of-sight velocity map, in km/s, with standard deviation
            set by `self.vlos_kw['std']`.
        """

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
        """
        Generate a velocity dispersion (sigma_v) map.

        Built from the same power-law field family as `get_velocity_field`
        (see `get_kinetic_powerlaw_field`), so that sigma_v shares its
        underlying spatial structure with v_los and (via that shared
        structure and `field_sign`) correlates or anti-correlates with column
        density. Default parameters mimic GAS & KEYSTONE NH3 results, where
        linewidth anti-correlates with column density.

        Parameters
        ----------
        seed : int, optional
            Random seed for field generation. If None, reuses `self.seed`.

        Returns
        -------
        field_sigv : ndarray
            2D velocity dispersion map, in km/s, log-normal with parameters
            set by `self.sigv_kw`.
        """
        field_pl = self.get_kinetic_powerlaw_field(seed=seed)
        field_pl *= self.sigv_kw['field_sign']  # correlate or anti-correlate with column density

        # Scale and exponentiate to create log-normal distribution for sigma_v
        mean = self.sigv_kw['mean_log']
        std = self.sigv_kw['std_log']
        field_scaled = np.log(mean) - 0.5 * (std ** 2) + std * field_pl
        self.field_sigv = np.exp(field_scaled)
        return self.field_sigv


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