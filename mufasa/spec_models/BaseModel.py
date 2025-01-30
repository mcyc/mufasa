"""
Base class for multi-component spectral models.

Provides core functionality for generating molecular line spectra with or without hyperfine structure.
"""

import numpy as np
from pyspeckit.spectrum.models import model
from astropy import constants
from astropy import units as u

# Universal constants
TCMB = 2.7315  # Cosmic Microwave Background temperature in K
h = constants.h.cgs.value  # Planck's constant in erg·s.
kb = constants.k_B.cgs.value  # Boltzmann constant in erg/K.

class BaseModel:
    """
    Generalized base class for multi-component spectral models.
    """

    _molecular_constants = None  # This is intended to be set in subclasses

    # Universal constants
    _TCMB = TCMB # Cosmic Microwave Background (CMB) temperature in Kelvin.
    _ckms = constants.c.to(u.km / u.s).value  # Speed of light in kilometers per second (km/s).
    _ccms = constants.c.to(u.cm / u.s).value  # Speed of light in centimeters per second (cm/s).
    _h = h  # Planck's constant in erg·s.
    _kb = kb  # Boltzmann constant in erg/K.

    def __init__(self, line_names=None):
        """
        Initialize the BaseModel with molecule-specific constants.

        Parameters
        ----------
        line_names : list of str, optional
            List of line names for the molecule. If not provided, use the default.
        """
        self.line_names = line_names or ['default_line']


    def multi_v_model_generator(self, n_comp):
        """
        Generate a multi-component spectral model.

        Parameters
        ----------
        n_comp : int
            Number of velocity components.

        Returns
        -------
        model : `model.SpectralModel`
            Spectral model with `n_comp` velocity components.
        """

        n_para = n_comp * 4  # vel, width, tex, tau per component
        idx_comp = np.arange(n_comp)

        nlines = len(self.line_names)
        if nlines > 1:
            raise NotImplementedError("Modeling more than one line is not yet implemented. Use a single line.")

        def vtau_multimodel(xarr, *args):
            assert len(args) == n_para
            return self.multi_v_spectrum(xarr, *args)

        mod = model.SpectralModel(
            vtau_multimodel, n_para,
            parnames=[x
                      for ln in idx_comp
                      for x in (f'vlsr{ln}', f'sigma{ln}', f'tex{ln}', f'tau{ln}')],
            parlimited=[(False, False), (True, False), (True, False), (True, False)] * n_para,
            parlimits=[(0, 0), ] * n_para,
            shortvarnames=[x
                           for ln in idx_comp
                           for x in (f'v_{{VLSR,{ln}}}', f'\\sigma_{{{ln}}}', f'T_{{ex,{ln}}}', f'\\tau_{{{ln}}}')],
            fitunit='Hz'
        )
        return mod


    def multi_v_spectrum(self, xarr, *args):
        """
        Generate a multi-component spectrum.

        Parameters
        ----------
        xarr : array-like
            Frequency array in GHz.
        args : list
            Model parameters (velocity, width, excitation temperature, optical depth)
            for each component, provided in sequence.

        Returns
        -------
        spectrum : array-like
            Computed spectrum for the given components.
        """
        cls = self.__class__

        if xarr.unit.to_string() != 'GHz':
            xarr = xarr.as_unit('GHz')

        background_ta = cls.T_antenna(cls.TCMB, xarr.value)
        tau_dict = {}

        for vel, width, tex, tau in zip(args[::4], args[1::4], args[2::4], args[3::4]):
            for linename in self.line_names:
                tau_dict[linename] = tau

            model_spectrum = self._single_spectrum(
                xarr, tex, tau_dict, width, vel, background_ta=background_ta
            )

            # Update background for the next component
            background_ta = model_spectrum

        return model_spectrum - cls.T_antenna(cls.TCMB, xarr.value)


    def _single_spectrum(self, xarr, tex, tau_dict, width, xoff_v, background_ta=0.0):
        """
        Compute a single-component molecular line spectrum.

        Parameters
        ----------
        xarr : array-like
            Frequency array in GHz.
        tex : float
            Excitation temperature (K).
        tau_dict : dict
            Optical depth for each transition.
        width : float
            Line width (km/s).
        xoff_v : float
            Velocity offset (km/s).
        background_ta : float or array-like, optional
            Background antenna temperature (default: 0.0).

        Returns
        -------
        spectrum : array-like
            Computed molecular line spectrum without hyperfine structure.
        """
        cls = self.__class__

        runspec = np.zeros(len(xarr))

        for linename in self.line_names:
            # Get molecule-specific constants
            freq_dict = cls._molecular_constants['freq_dict']

            # Retrieve the central frequency for the given transition
            line = freq_dict[linename] / 1e9  # Convert to GHz

            # Compute single-value quantities (no hyperfine structure)
            nuoff = xoff_v / cls._ckms * line  # Shift frequency by velocity offset
            nuwidth = np.abs(width / cls._ckms * line)  # Compute Gaussian width
            tau0 = tau_dict[linename]  # Optical depth for this transition

            # Compute the optical depth profile (single Gaussian function)
            tauprof = tau0 * np.exp(-((xarr.value + nuoff - line) ** 2) / (2.0 * nuwidth ** 2))

            # Compute Planck function temperature
            T0 = (cls._h * xarr.value * 1e9 / cls._kb)

            # Compute the spectrum
            runspec += (T0 / (np.exp(T0 / tex) - 1) * (1 - np.exp(-tauprof)) +
                        background_ta * np.exp(-tauprof))

        return runspec  # Return the molecular line spectrum

    @staticmethod
    def T_antenna(Tbright, nu):
        """
        Compute the antenna temperature.

        Parameters
        ----------
        Tbright : float
            Brightness temperature (K).
        nu : array-like
            Frequency array in GHz.

        Returns
        -------
        T_antenna : array-like
            Computed antenna temperature.
        """
        T0 = (h * nu * 1e9 / kb)
        return T0 / (np.exp(T0 / Tbright) - 1)
