
__author__ = 'mcychen'

import numpy as np
from astropy import units as u
from concurrent.futures import ThreadPoolExecutor

from ..exceptions import FitTypeError
from .. import moment_guess as momgue


class MetaModel(object):
    """
    A class to store spectral model-specific information relevant to spectral modeling tasks, such as fitting.

    Attributes
    ----------
    fittype : str
        The identifying name for the spectral model (i.e., the type of model to fit).
    ncomp : int
        The number of components within the model.
    adoptive_snr_masking : bool
        Whether to use adaptive SNR masking during fitting (default is True).
    window_hwidth : int
        The default half-width of the window for calculating moments.
    central_win_hwidth : float or None
        The half-width of the intermediate window for filtering out satellite hyperfine lines. Default is None.
    Texmin : float
        Lower excitation temperature limit in Kelvin. Default is 3.0 K.
    Texmax : float
        Upper excitation temperature limit in Kelvin. Default is 40 K.
    sigmin : float
        Lower velocity dispersion limit in km/s. Default is 0.07 km/s.
    sigmax : float
        Upper velocity dispersion limit in km/s. Default is 2.5 km/s.
    taumin : float
        Lower optical depth limit. Default is 0.1.
    taumax : float
        Upper optical depth limit. Default is 30.0.
    eps : float
        A small perturbation that can be used in initial guesses. Default is 0.001.
    main_hf_moments : function
        Function for calculating moments for the main hyperfine component.
    linetype : str
        The type of spectral line (e.g., 'nh3', 'n2hp'). Set based on `fittype`.
    linenames : list of str
        Names of the spectral lines being modeled. Set based on `fittype`.
    linename : str
        Name of the main spectral line being modeled. Set based on `fittype`.
    fitter : function
        The model generator function for fitting spectral components. Set based on `fittype`.
    freq_dict : dict
        Dictionary containing rest frequencies for the spectral lines. Set based on `fittype`.
    rest_value : astropy.units.Quantity
        Rest frequency of the spectral line in Hz. Set based on `fittype`.
    voffs : list of float
        Velocity offsets for the hyperfine lines in km/s. Set based on `fittype`.
    tau_wts : list of float
        Weights for the optical depth of hyperfine lines. Set based on `fittype`.
    voff_at_peak : float
        Velocity offset at the brightest hyperfine line in km/s. Set based on `fittype`.
    model_func : function
        The specific spectral model function used for fitting. Set based on `fittype`.
    """

    def __init__(self, fittype, ncomp):
        """
        Initialize a MetaModel object.

        Parameters
        ----------
        fittype : str
            The identifying name for the spectral model (e.g., 'nh3_multi_v', 'n2hp_multi_v').
        ncomp : int
            The number of components within the model to be fitted.

        Raises
        ------
        FitTypeError
            If the provided `fittype` is not recognized.
        """
        self.fittype = fittype
        self.adoptive_snr_masking = True

        # the default windows for calculating moments
        # final window for evaluating moments (center differs from pixel to pixel)
        self.window_hwidth = 5
        # intermediate window for filtering out satellite hyperfine lines. The window is fixed and the same
        # throughout all pixels
        self.central_win_hwidth = None

        # set default parameter limits
        # (primarily based on NH3 (1,1) in the galactic molecular cloud conditions using RADEX)
        self.Texmin = 3.0  # K; (assume T_kin=5K, density = 1e3 cm^-3, column=1e13 cm^-2, sigv=3km/s)
        self.Texmax = 40  # K; T_k for Orion A is < 35 K. (T_k=40 at 1e5 cm^-3, 1e15 cm^-2, and 0.1 km/s --> Tex=37K)
        self.sigmin = 0.07  # km/s
        self.sigmax = 2.5  # km/s; (Larson's law for a 10pc cloud has sigma = 2.6 km/s)
        self.taumax = 30.0  # thick enough for NH3 (1,1) satellite hyperfines to have the same intensity as the main
        self.taumin = 0.1  # (note: NH3 (1,1) at 1e3 cm^-3, 1e13 cm^-2, 1 km/s linewidth, 40 K -> 0.15)
        self.eps = 0.001  # a small perturbation that can be used in guesses

        # a placeholder for if the window becomes line specific
        self.main_hf_moments = momgue.window_moments

        # for the NH3 (1,1) multicomponent model
        if self.fittype == 'nh3_multi_v':
            from pyspeckit.spectrum.models.ammonia_constants import freq_dict, voff_lines_dict, tau_wts_dict
            from . import ammonia_multiv as ammv
            from importlib import reload
            reload(ammv)

            self.linetype = 'nh3'
            # the current implementation only fits the 1-1 lines
            self.linename = 'oneone'
            self.linenames = [self.linename]
            self.fitter = ammv.nh3_multi_v_model_generator(n_comp=ncomp, linenames=self.linenames)

            self.freq_dict = freq_dict
            self.rest_value = freq_dict[self.linename] * u.Hz
            self.voffs = voff_lines_dict[self.linename]
            self.tau_wts = tau_wts_dict[self.linename]
            self.voff_at_peak = self.voffs[np.argmax(self.tau_wts)] # velocity offset at the brightest hyperfine line

            # the function to compute the spectral model
            self.model_func = ammv.ammonia_multi_v


        # for the N2H+ (1-0) multi-component model
        elif self.fittype == 'n2hp_multi_v':
            from .n2hp_constants import freq_dict, voff_lines_dict, tau_wts_dict
            from . import n2hp_multiv as n2hpmv

            self.linetype = 'n2hp'
            # the current implementation only fits the 1-0 lines
            self.linename = 'onezero'
            self.linenames = [self.linename]
            self.fitter = n2hpmv.n2hp_multi_v_model_generator(n_comp=ncomp, linenames=self.linenames)

            self.freq_dict = freq_dict
            self.rest_value = freq_dict[self.linename] * u.Hz
            self.voffs = voff_lines_dict[self.linename]
            self.tau_wts = tau_wts_dict[self.linename]
            self.voff_at_peak = self.voffs[np.argmax(self.tau_wts)] # velocity offset at the brightest hyperfine line

            # the function to compute the spectral model
            self.model_func = n2hpmv.n2hp_multi_v

            # change the parameter limits from the default to better reflects N2H+ (1-0)
            self.taumax = 40.0  # when the satellite hyperfine lines becomes optically thick

            # overwrite default windows, since N2H+ (1-0) have a satellite line that could be fairly bright
            self.window_hwidth = 4
            self.central_win_hwidth = 3.5

        else:
            raise FitTypeError("\'{}\' is an invalid fittype".format(fittype))


    def model_function(self, xarr, parameters, planemask=None, multithreaded=False):
        """
        Compute the spectral model for a given spectral axis and model parameters.

        Parameters
        ----------
        xarr : pyspeckit.spectrum.units.SpectroscopicAxis
            The spectral axis along which to evaluate the model function.
        parameters : array-like
            The model parameters for evaluation. Can be:
            - A 1D array with shape (l,), representing a single set of parameters.
            - A 2D array with shape (l, n), where `l` is the number of parameters, and `n` is the number of spatial positions.
            - A 3D array with shape (l, m, n), where `l` is the number of parameters per pixel, and `m` and `n` represent
              the spatial dimensions. In this case, the function will evaluate the model at each (m, n) pixel specified by `planemask`.
        planemask : array-like, optional
            A 2D mask array with shape (m, n) indicating the pixels to evaluate when `parameters` is 3D.
            Pixels with `True` in the mask will be evaluated. If None, a default mask is generated where
            all non-zero, finite values in `parameters` are considered valid.
        multithreaded : bool, optional
            Whether to enable multithreading to evaluate pixels in parallel when `parameters` is 2D or 3D.
            Default is False.

        Returns
        -------
        np.ndarray
            The evaluated spectral model values. The output shape matches the input shape of `parameters`:
            - If `parameters` is 1D, returns a 1D array of computed values.
            - If `parameters` is 2D, returns a 1D array with length `n`, containing computed values for each position.
            - If `parameters` is 3D, returns a 2D array with shape (m, n), containing computed values for each
              evaluated pixel in the masked region. Masked pixels are filled with NaN.
        """

        if parameters.ndim == 1:
            # Direct evaluation for 1D input
            return self.model_func(xarr, *parameters)

        elif parameters.ndim == 2:
            # For 2D input, transpose for easier iteration
            pars = parameters.T

            # Initialize results array for 2D output
            results = np.full(pars.shape[0], np.nan, dtype=object)

            # Define evaluation function for each position
            def evaluate_position(i):
                par = pars[i]
                return self.model_func(xarr, *par)

            if multithreaded:
                with ThreadPoolExecutor() as executor:
                    evaluated_results = list(executor.map(evaluate_position, range(len(pars))))
                results[:] = evaluated_results
            else:
                for i, par in enumerate(pars):
                    results[i] = self.model_func(xarr, *par)

            return results

        elif parameters.ndim == 3:
            # Generate mask if not provided
            if planemask is None:
                planemask = np.all(np.isfinite(parameters), axis=0) & np.all(parameters != 0, axis=0)

            # Mask parameters for efficient evaluation
            pars = parameters[:, planemask].T

            # Initialize results array to hold model outputs for 3D input
            results = np.full((parameters.shape[1], parameters.shape[2]), np.nan, dtype=object)

            # Define evaluation function for each pixel
            def evaluate_pixel(i):
                par = pars[i]
                return self.model_func(xarr, *par)

            if multithreaded:
                with ThreadPoolExecutor() as executor:
                    evaluated_results = list(executor.map(evaluate_pixel, range(len(pars))))
                results[planemask] = evaluated_results
            else:
                for i, par in enumerate(pars):
                    results[planemask][i] = self.model_func(xarr, *par)

            return results


    def peakT(self, parameters, index_v=0, planemask=None, multithreaded=False):
        """
        Estimate the peak brightness temperature of the spectral model by evaluating the model at the spectral location
        (i.e., the velocity) of the brightest hyperfine line. Note: this method only works for a single-component model,
        where the peak brightness is expected to be located near the brightest hyperfine line.

        Parameters
        ----------
        parameters : array-like
            The model parameters to evaluate, where each element corresponds to a parameter used in the spectral model.
            Can be:
            - A 1D array with shape (l,), representing a single set of parameters.
            - A 2D array with shape (l, n), where `l` is the number of parameters and `n` is the number of spatial positions.
            - A 3D array with shape (l, m, n), where `l` is the number of parameters per pixel, and `m` and `n` represent
              the spatial dimensions.
        index_v : int, optional
            Index of the parameter corresponding to the velocity centroid, which is 0 for all spectral models
            implemented in MUFASA. Default is 0.
        planemask : array-like, optional
            A 2D mask array with shape (m, n) indicating the pixels to evaluate when `parameters` is 3D.
            Pixels with `True` in the mask will be evaluated. If None, a mask is generated based on finite, non-zero values
            in `parameters`.
        multithreaded : bool, optional
            Whether to enable multithreading to evaluate pixels in parallel when `parameters` is 2D or 3D.
            Default is False.

        Returns
        -------
        np.ndarray
            An array with the peak brightness temperature for each evaluated pixel:
            - If `parameters` is 1D, returns a scalar representing the peak brightness temperature.
            - If `parameters` is 2D, returns a 1D array with length `n`, containing the peak brightness temperature for each
              position.
            - If `parameters` is 3D, returns a 2D array with shape (m, n) containing the peak brightness temperature for each
              evaluated pixel in the masked region. Masked pixels are filled with NaN.
        """

        # Function implementation

        # Get the velocity at the brightest hyperfine line
        v0 = np.array([parameters[index_v] + self.voff_at_peak]) * u.km / u.s

        # Convert the velocity to frequency for evaluation
        nu0 = v0.to(u.GHz, equivalencies=u.doppler_radio(self.rest_value))

        if parameters.ndim == 1:
            # Direct evaluation for 1D input
            return self.model_func(nu0, *parameters)[0]

        else:
            if parameters.ndim == 2:
                pars = parameters.T
                nu0_array = nu0.T

            if parameters.ndim == 3:
                # Generate mask if not provided
                if planemask is None:
                    planemask = np.all(np.isfinite(parameters), axis=0) & np.all(parameters != 0, axis=0)

                # Mask parameters and frequencies for efficient evaluation
                pars = parameters[:, planemask].T
                nu0_array = nu0[:, planemask].T

            # Initialize results array
            pT = np.full(nu0_array.shape[0], np.nan)

            # Define a function for multithreading
            def evaluate_pixel(i):
                nu0_i = nu0_array[i]
                par = pars[i]
                return self.model_func(nu0_i, *par)[0]

            if multithreaded:
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(evaluate_pixel, range(len(pars))))
                pT[:] = results
            else:
                # Single-threaded loop if multicore is False
                for i, par in enumerate(pars):
                    nu0_i = nu0_array[i]
                    pT[i] = self.model_func(nu0_i, *par)[0]

            if parameters.ndim == 3:
                # Reconstruct the 2D output
                pT2D = np.full(planemask.shape, np.nan)
                pT2D[planemask] = pT
                return pT2D

            else:
                return pT
