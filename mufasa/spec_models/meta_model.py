
__author__ = 'mcychen'

from astropy import units as u

from ..exceptions import FitTypeError
from .. import moment_guess as momgue


class MetaModel(object):
    """
    A class to host spectral-model-specific information that is relevant to spectral model tasks, such as fitting.

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
        Lower excitation limit temperature in Kelvin. Default is 3.0 K.
    Texmax : float
        Upper excitation limit temperature in Kelvin. Default is 40 K.
    sigmin : float
        Lower velocity limit dispersion in km/s. Default is 0.07 km/s.
    sigmax : float
        Upper velocity limit dispersion in km/s. Default is 2.5 km/s.
    taumin : float
        Lower optical depth limit. Default is 0.1.
    taumax : float
        Upper optical depth limit. Default is 30.0.
    eps : float
        A small perturbation that can be used in guesses. Default is 0.001.
    main_hf_moments : function
        Function for calculating moments for the main hyperfine component.
    linetype : str
        The type of spectral line (e.g., 'nh3', 'n2hp'). Set based on fittype.
    linenames : list of str
        Names of the spectral lines being modeled. Set based on fittype.
    fitter : function
        The model generator function for fitting spectral components. Set based on fittype.
    freq_dict : dict
        Dictionary containing rest frequencies for the spectral lines. Set based on fittype.
    rest_value : astropy.units.Quantity
        Rest frequency of the spectral line in Hz. Set based on fittype.
    spec_model : function
        The specific spectral model function used for fitting. Set based on fittype.
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
        # intermediate window for filtering out satallite hyperfine lines. The window is fixed and the same
        # throughout all pixels
        self.central_win_hwidth = None

        # set default parameter limits
        # (primarily based on NH3 (1,1) in the galatic molecular cloud conditions using RADEX)
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
            from pyspeckit.spectrum.models.ammonia_constants import freq_dict
            from pyspeckit.spectrum.models import ammonia
            from . import ammonia_multiv as ammv

            self.linetype = 'nh3'
            # the current implementation only fits the 1-1 lines
            self.linenames = ["oneone"]

            self.fitter = ammv.nh3_multi_v_model_generator(n_comp=ncomp, linenames=self.linenames)
            self.freq_dict = freq_dict
            self.rest_value = freq_dict['oneone'] * u.Hz
            self.spec_model = ammonia._ammonia_spectrum

        # for the N2H+ (1-0) multi-component model
        elif self.fittype == 'n2hp_multi_v':
            from .n2hp_constants import freq_dict
            from . import n2hp_multiv as n2hpmv

            self.linetype = 'n2hp'
            # the current implementation only fits the 1-0 lines
            self.linenames = ["onezero"]

            self.fitter = n2hpmv.n2hp_multi_v_model_generator(n_comp=ncomp, linenames=self.linenames)
            self.freq_dict = freq_dict
            self.rest_value = freq_dict['onezero'] * u.Hz
            self.spec_model = n2hpmv._n2hp_spectrum

            # change the parameter limits from the default to better reflects N2H+ (1-0)
            self.taumax = 40.0  # when the satellite hyperfine lines becomes optically thick

            # overwrite default windows, since N2H+ (1-0) have a satellite line that could be fairly bright
            self.window_hwidth = 4
            self.central_win_hwidth = 3.5

        else:
            raise FitTypeError("\'{}\' is an invalid fittype".format(fittype))
