from __future__ import print_function
import numpy as np
from pyspeckit.spectrum.models import model
from pyspeckit.spectrum.models.ammonia import (ckms, h, kb)

class BaseModel:
    """
    Generalized base class for multi-component spectral models.

    Attributes
    ----------
    molecular_constants : dict
        Dictionary containing molecule-specific parameters, such as:
            - freq_dict
            - voff_lines_dict
            - tau_wts_dict
    line_names : list of str
        Default line names for the molecule (e.g., ['oneone'], ['onezero']).
    """
    molecular_constants = None
    line_names = None

    # Universal constants
    TCMB = 2.7315  # Cosmic Microwave Background temperature in K
    ckms = ckms  # Speed of light in km/s
    h = h        # Planck constant
    kb = kb      # Boltzmann constant

    def __init__(self, molecular_constants, line_names=None):
        """
        Initialize the BaseModel with molecule-specific constants.

        Parameters
        ----------
        molecular_constants : dict
            Molecule-specific constants (freq_dict, voff_lines_dict, tau_wts_dict).
        line_names : list of str, optional
            List of line names for the molecule. If not provided, use the default.
        """
        if not molecular_constants:
            raise ValueError("Molecular constants (freq_dict, voff_lines_dict, tau_wts_dict) must be provided.")
        self.molecular_constants = molecular_constants
        self.line_names = line_names or ['default_line']

    @classmethod
    def multi_v_model_generator(cls, n_comp):
        """
        Generalized model generator for multi-component spectral models.

        Parameters
        ----------
        n_comp : int
            Number of velocity components to fit.

        Returns
        -------
        model : `model.SpectralModel`
            A spectral model class built from N different velocity components.
        """
        n_para = n_comp * 4  # vel, width, tex, tau per component
        idx_comp = np.arange(n_comp)

        nlines = len(cls.line_names)
        if nlines > 1:
            raise NotImplementedError("Modeling more than one line is not yet implemented. Use a single line.")

        def vtau_multimodel(xarr, *args):
            assert len(args) == n_para
            return cls.multi_v_spectrum(xarr, *args)

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

    @classmethod
    def multi_v_spectrum(cls, xarr, *args):
        """
        Generalized multi-component spectrum generator.

        Parameters
        ----------
        xarr : array-like
            Frequency array (in GHz).
        args : list
            Model parameters (vel, width, tex, tau) for each component.

        Returns
        -------
        spectrum : array-like
            Generated spectrum for the given parameters.
        """
        if xarr.unit.to_string() != 'GHz':
            xarr = xarr.as_unit('GHz')

        molecular_constants = cls.molecular_constants
        freq_dict = molecular_constants['freq_dict']
        voff_lines_dict = molecular_constants['voff_lines_dict']
        tau_wts_dict = molecular_constants['tau_wts_dict']

        background_ta = cls.T_antenna(cls.TCMB, xarr.value)
        tau_dict = {}

        for vel, width, tex, tau in zip(args[::4], args[1::4], args[2::4], args[3::4]):
            for linename in cls.line_names:
                tau_dict[linename] = tau

            model_spectrum = cls._single_spectrum(
                xarr, tex, tau_dict, width, vel, background_ta=background_ta
            )

            # Update background for the next component
            background_ta = model_spectrum

        return model_spectrum - cls.T_antenna(cls.TCMB, xarr.value)

    @classmethod
    def _single_spectrum(cls, xarr, tex, tau_dict, width, xoff_v, background_ta=0.0):
        """
        Generalized helper function to compute single-component spectrum.

        Parameters
        ----------
        xarr : array-like
            Frequency array (in GHz).
        tex : float
            Excitation temperature.
        tau_dict : dict
            Optical depth dictionary.
        width : float
            Line width (in km/s).
        xoff_v : float
            Velocity offset (in km/s).
        background_ta : float or array-like
            Background antenna temperature.

        Returns
        -------
        spectrum : array-like
            Generated single-component spectrum.
        """
        molecular_constants = cls.molecular_constants
        freq_dict = molecular_constants['freq_dict']
        voff_lines_dict = molecular_constants['voff_lines_dict']
        tau_wts_dict = molecular_constants['tau_wts_dict']

        runspec = np.zeros(len(xarr))
        for linename in cls.line_names:
            voff_lines = np.array(voff_lines_dict[linename])
            tau_wts = np.array(tau_wts_dict[linename])
            lines = (1 - voff_lines / cls.ckms) * freq_dict[linename] / 1e9
            tau_wts /= tau_wts.sum()
            nuwidth = np.abs(width / cls.ckms * lines)
            nuoff = xoff_v / cls.ckms * lines

            tauprof = np.zeros(len(xarr))
            for kk, nuo in enumerate(nuoff):
                tauprof += (tau_dict[linename] * tau_wts[kk] *
                            np.exp(-(xarr.value + nuo - lines[kk]) ** 2 /
                                   (2.0 * nuwidth[kk] ** 2)))

            T0 = (cls.h * xarr.value * 1e9 / cls.kb)
            runspec += ((T0 / (np.exp(T0 / tex) - 1) * (1 - np.exp(-tauprof)) +
                         background_ta * np.exp(-tauprof)))

        return runspec

    @staticmethod
    def T_antenna(Tbright, nu):
        """
        Calculate antenna temperatures over frequency (GHz).

        Parameters
        ----------
        Tbright : float
            Brightness temperature.
        nu : array-like
            Frequency array (in GHz).

        Returns
        -------
        T_antenna : array-like
            Antenna temperature.
        """
        T0 = (h * nu * 1e9 / kb)
        return T0 / (np.exp(T0 / Tbright) - 1)
