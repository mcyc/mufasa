
import numpy as np
from .BaseModel import BaseModel

class HyperfineModel(BaseModel):
    """
    A subclass of BaseModel generate spectral models with hyperfine lines.

    Overrides the multi_v_spectrum() from BaseModel to use hyperfine lines

    """

    def __init__(self, line_names=None):
        """
        Initialize HyperfineModel.

        """
        super().__init__(line_names=line_names)


    def multi_v_spectrum(self, xarr, *args):
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
        cls = self.__class__

        if xarr.unit.to_string() != 'GHz':
            xarr = xarr.as_unit('GHz')

        background_ta = cls.T_antenna(cls.TCMB, xarr.value)
        tau_dict = {}

        for vel, width, tex, tau in zip(args[::4], args[1::4], args[2::4], args[3::4]):
            for linename in self.line_names:
                tau_dict[linename] = tau

            model_spectrum = self._single_spectrum_hf(
                xarr, tex, tau_dict, width, vel, background_ta=background_ta
            )

            # Update background for the next component
            background_ta = model_spectrum

        return model_spectrum - cls.T_antenna(cls.TCMB, xarr.value)


    def _single_spectrum_hf(self, xarr, tex, tau_dict, width, xoff_v, background_ta=0.0):
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
        cls = self.__class__
        molecular_constants = cls.molecular_constants
        freq_dict = molecular_constants['freq_dict']
        voff_lines_dict = molecular_constants['voff_lines_dict']
        tau_wts_dict = molecular_constants['tau_wts_dict']

        runspec = np.zeros(len(xarr))
        for linename in self.line_names:
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


    def deblend(self, xarr, tex, tau, xoff_v, width):
        return self._single_spectrum(xarr, tex, tau, width, xoff_v, background_ta=0.0)




