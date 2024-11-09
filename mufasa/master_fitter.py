from __future__ import print_function
from __future__ import absolute_import

__author__ = 'mcychen'

# =======================================================================================================================
import os
import numpy as np
import multiprocessing
from spectral_cube import SpectralCube
from astropy import units as u
from skimage.morphology import binary_dilation, remove_small_holes, remove_small_objects, square, disk
from skimage.filters import threshold_local
import astropy.io.fits as fits
from copy import copy, deepcopy
import gc
from scipy.signal import medfilt2d
from time import ctime
from datetime import timezone, datetime
import warnings
import pandas as pd
from datetime import datetime

import dask.array as da

try:
    from pandas import concat
except ImportError:
    # this will only be an issue if the padas version is very old
    pass

from . import UltraCube as UCube
from . import moment_guess as mmg
from . import convolve_tools as cnvtool
from . import guess_refine as gss_rf
from .exceptions import SNRMaskError, FitTypeError, StartFitError
from .utils.multicore import validate_n_cores
from .utils import neighbours
# =======================================================================================================================
from .utils.mufasa_log import init_logging, get_logger

logger = get_logger(__name__)


# =======================================================================================================================

class Region(object):
    """
    A class to represent a region of a spectral cube and perform analysis on it.

    Attributes
    ----------
    cubePath : str
        Path to the spectral cube FITS file.
    paraNameRoot : str
        Root string used for naming output files.
    paraDir : str
        Directory to store output files.
    fittype : str
        Type of fitting to use.
    ucube : UCubePlus
        UltraCube object for handling spectral cube operations.
    cnv_factor : int
        Number of beam-widths to convolve by.
    progress_log_name : str
        Path to the progress log CSV file.
    progress_log : pandas.DataFrame
        DataFrame to log the progress of different processes.
    """


    def __init__(self, cubePath, paraNameRoot, paraDir=None, cnv_factor=2, fittype=None, initialize_logging=True,
                 multicore=True, **kwargs):
        """
        Initialize region object

        Parameters
        ----------
        cubePath : str
            Path to the spectral cube FITS file.
        paraNameRoot : str
            A common root string to name the output .fits files.
        paraDir : str, optional
            Directory to store output files (default is None, in which case outputs are saved in the same directory as the input cube).
        cnv_factor : int, optional
            The factor spatially convolve the cube if needed by the interative fitting process (default is 2).
        fittype : str, optional
            The spectral model to use for the fit (default is 'nh3_multi_v').
        initialize_logging : bool, optional
            Whether to initialize logging (default is True).
        multicore : bool or int, optional
            Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
            If an integer is provided, it specifies the number of CPU cores to use.

        **kwargs : dict, optional
        Additional keyword arguments passed to the logging initialization.

            :param cubePath (str): path to spectral cube
            :param paraNameRoot (str): string to prepend to output file names
            :param paraDir (str, optional): directory to output files. Defaults to None.
            :param cnv_factor (int, optional): nuber of beam-widths to convolve by. Defaults to 2.
            :param multicore (int, optional): number of cpu cores to use. Defaults to None.

            kwargs are passed to init_logging:
                :param logfile: file to save to (default mufasa.log)
                :param console_level: minimum logging level to print to screen (default logging.INFO)
                :param file_level: minimum logging level to save to file (default logging.INFO)
                :param log_pyspeckit_to_file: whether to include psypeckit outputs in log file (default False)
        """

        if initialize_logging: init_logging(**kwargs)
        self.cubePath = cubePath
        self.paraNameRoot = paraNameRoot
        self.paraDir = paraDir

        if fittype is None:
            fittype = 'nh3_multi_v'
            message = "[WARNING] The optionality of the fittype argment for the Region class will be deprecated in the future. " \
                      "Please ensure the fittype argument is specified going forward."
            warnings.warn(message, DeprecationWarning, stacklevel=2)

        self.fittype = fittype
        self.ucube = UCube.UCubePlus(cubePath, paraNameRoot=paraNameRoot, paraDir=paraDir, cnv_factor=cnv_factor,
                                     fittype=self.fittype, n_cores=multicore)

        # for convolving cube
        self.cnv_factor = cnv_factor
        self.progress_log_name = "{}/{}_progress_log.csv".format(self.ucube.paraDir, self.ucube.paraNameRoot)

        self.timestemp = None
        try:
            from pandas import read_csv
            self.progress_log = read_csv(self.progress_log_name)
            # could use a checking mechanism to ensure the logfile is the right format
        except FileNotFoundError:
            columns = ['process', 'last completed', 'attempted fits (pix)', 'successful fits (pix)',
                       'total runtime (dd:hh:mm:ss)', 'cores used']
            self.progress_log = pd.DataFrame(columns=columns)


    def get_convolved_cube(self, update=True, cnv_cubePath=None, edgetrim_width=5, paraNameRoot=None, paraDir=None,
                           multicore=True):
        if paraDir is None:
            paraDir = self.paraDir
        get_convolved_cube(self, update=update, cnv_cubePath=cnv_cubePath, edgetrim_width=edgetrim_width,
                           paraNameRoot=paraNameRoot, paraDir=paraDir, multicore=multicore)

    def get_convolved_fits(self, ncomp, **kwargs):
        get_convolved_fits(self, ncomp, **kwargs)

    def get_fits(self, ncomp, **kwargs):
        get_fits(self, ncomp, **kwargs)

    def load_fits(self, ncomp):
        # basically the same as get_fits(), but with update set to False to ensure the fits aren't refitted
        get_fits(self, ncomp, update=False)

    def master_2comp_fit(self, snr_min=0.0, **kwargs):
        master_2comp_fit(self, snr_min=snr_min, **kwargs)

    def standard_2comp_fit(self, planemask=None):
        standard_2comp_fit(self, planemask=planemask)

    def read_best_fits(self, ncomp=2, header=True):
        """
        Read the saved .fits file containing the fitted parameters for the best models up to `ncomp` components, output by MUFASA.

        Parameters
        ----------
        ncomp : int, optional
            The maximum number of components in the model to read. Default is 2.
        header : bool, optional
            If True, includes the header information from the .fits file in the output. Default is True.

        Returns
        -------
        data : numpy.ndarray
            The data array containing the fitted parameters from the .fits file.
        hdr : astropy.io.fits.Header, optional
            The header of the .fits file, included if `header` is True.

        Notes
        -----
        This method constructs the file path based on the specified number of components (`ncomp`) and reads the .fits file
        generated by MUFASA, which contains the fitted model parameters. If `header` is True, both the data and header are returned.
        """

        filepath = "{}_final.fits".format(os.path.splitext(self.ucube.paraPaths[str(ncomp)])[0])
        return fits.getdata(filepath, header=header)

    def plot_ppv_scatter(self, ncomp=2, label_key='peakT', savename=None, vel_scale=0.8, vrange=None, verr_thres=5,
                         **kwargs):
        """
        Generate a 3D scatter plot of the position-position-velocity (PPV) data from MUFASA generaged fitted parameters for the
        best model, up to 'ncomp' of components.

        Parameters
        ----------
        ncomp : int, optional
            The number of components in the model to be plotted.
        label_key : str, optional
            DataFrame column to color each data point by, e.g., 'peakT' for peak intensity or clustering labels. Default is 'peakT'.
        savename : str, optional
            The file path to save the 3D plot as an HTML file. If None, the plot will not be saved. Default is None.
        vel_scale : float, optional
            Scale factor for the velocity axis relative to the x and y axes, with x normalized to 1. Default is 0.8.
        vrange : tuple of float, optional
            Velocity range to clip the data (in km/s).
        verr_thres : float, optional
            The velocity error threshold (in km/s) to filter the data. Data with errors above this threshold is excluded.
        kwargs : dict, optional
            Additional keyword arguments are passed to `plot_ppv`, allowing customization of the PPV scatter plot.

            Key options include:
            - `xyunit` : {'arcmin', 'pix'}, optional
                Units for x & y coordinates, either in arcminutes or pixels. Default is 'arcmin'.
            - `auto_open_html` : bool, optional
                If True, automatically opens the saved HTML file in a browser after saving. Default is True.

            For a complete list of options, refer to `plot_ppv`.

        Returns
        -------
        plotly.graph_objs.Figure
            The generated 3D scatter plot figure.

        Notes
        -----
        This method initializes a `ScatterPPV` instance using the FITS file specified by `self.ucube.paraPaths` for the
        selected component count (`ncomp`). It then calls the `plot_ppv` method to generate the PPV scatter plot, allowing
        for optional customization through `savename`, `vel_scale`, and `kwargs`.
        """

        from .visualization import scatter_3D

        try:
            filepath = "{}_final.fits".format(os.path.splitext(self.ucube.paraPaths[str(ncomp)])[0])
        except KeyError:
            filepath = '{}/{}_{}vcomp_final.fits'.format(self.ucube.paraDir, self.ucube.paraNameRoot, ncomp)

        self.ScatterPPV = scatter_3D.ScatterPPV(filepath, fittype=self.fittype, vrange=vrange, verr_thres=verr_thres)
        return self.ScatterPPV.plot_ppv(label_key=label_key, savename=savename, vel_scale=vel_scale, **kwargs)


    def log_progress(self, process_name, mark_start=False, save=True, timespec='seconds', n_attempted=None,
                     n_success=None, finished=False, cores=None):
        """
        Log the progress of a process with start and completion times, including the number of attempted and successful fits.

        Parameters
        ----------
        process_name : str
            Name of the process to log.
        mark_start : bool, optional
            If True, marks the start of the process. Default is False.
        save : bool, optional
            If True, saves the log to a CSV file. Default is True.
        timespec : str, optional
            Specifies the level of detail for the timestamp. Default is 'seconds'.
            Options include 'seconds', 'milliseconds', etc.
        n_attempted : int, optional
            Number of attempted fits, corresponds to 'attempted fits (pix)'. Default is None.
        n_success : int, optional
            Number of successful fits, corresponds to 'successful fits (pix)'. Default is None.
        finished : bool, optional
            If True and `mark_start` is False, records the time elapsed in 'total runtime (dd:hh:mm:ss)' and clears `self.timestamp`. Default is False.
        cores : int, optional
            Number of cores used, corresponds to 'cores used'. Default is None.

        Notes
        -----
        - When `mark_start` is True, the 'last completed' column is set to the current timestamp and 'total runtime (dd:hh:mm:ss)' is set to "in progress".
        - When `mark_start` is False and `finished` is True, the function calculates the runtime from the recorded start time (`self.timestamp`) and records it in a simplified format.
        - The `self.timestamp` attribute is cleared only if `mark_start` is False and `finished` is True.
        - If the DataFrame `self.progress_log` does not exist, it will be initialized with default columns.
        """
        if mark_start:
            # Mark the start time
            self.timestamp = datetime.now()
            time_info = self.timestamp.isoformat(timespec=timespec)
            runtime_info = "in progress"
        else:
            timestamp = datetime.now()
            time_info = timestamp.isoformat(timespec=timespec)

            # Calculate runtime if finished is True
            if self.timestamp is not None and finished:
                runtime_delta = timestamp - self.timestamp

                days = runtime_delta.days
                hours, remainder = divmod(runtime_delta.seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_info = ""
                if days > 0:
                    runtime_info += f"{days}d "
                if hours > 0:
                    runtime_info += f"{hours}h "
                if minutes > 0:
                    runtime_info += f"{minutes:02}:"
                if seconds < 10:
                    runtime_info += f"{seconds + runtime_delta.microseconds / 1e6:.2f}"
                elif seconds < 60:
                    runtime_info += f"{seconds + runtime_delta.microseconds / 1e6:.1f}"
                else:
                    runtime_info += f"{seconds}"
                # Clear the timestamp as the task is completed
                self.timestamp = None
            else:
                runtime_info = "in progress" if self.timestamp is not None else "N/A"

        # Initialize progress log if it doesn't exist yet
        if not hasattr(self, 'progress_log'):
            columns = ['process', 'last completed', 'attempted fits (pix)', 'successful fits (pix)',
                       'total runtime (dd:hh:mm:ss)', 'cores used']
            self.progress_log = pd.DataFrame(columns=columns)

        cores_info = int(cores) if cores is not None else "N/A"

        if process_name in self.progress_log['process'].values:
            # Replace the previous entry if the process has been logged
            mask = self.progress_log['process'] == process_name
            self.progress_log.loc[mask, ['last completed', 'total runtime (dd:hh:mm:ss)']] = [time_info, runtime_info]

            # Update cores used only if the existing value is the default "N/A"
            if cores is not None or self.progress_log.loc[mask, 'cores used'].values[0] == "N/A":
                self.progress_log.loc[mask, 'cores used'] = cores_info

            # Update attempted fits and successful fits if arguments are provided
            if n_attempted is not None:
                self.progress_log.loc[mask, 'attempted fits (pix)'] = n_attempted
            if n_success is not None:
                self.progress_log.loc[mask, 'successful fits (pix)'] = n_success
        else:
            # Add a new entry otherwise
            # Fill the additional columns with default values (e.g., "None" for unspecified fields)
            info = {
                'process': process_name,
                'last completed': time_info,
                'attempted fits (pix)': n_attempted if n_attempted is not None else "None",
                'successful fits (pix)': n_success if n_success is not None else "None",
                'total runtime (dd:hh:mm:ss)': runtime_info,
                'cores used': cores_info
            }

            if pd.__version__ >= '1.3.0':
                # For newer pandas versions, use concat
                self.progress_log = pd.concat([self.progress_log, pd.DataFrame([info])], ignore_index=True)
            else:
                # For older pandas versions, use append
                self.progress_log = self.progress_log.append(info, ignore_index=True)

        if save:
            # Reorder the rows by 'last completed' in chronological order
            self.progress_log = self.progress_log.sort_values(by='last completed', ascending=True)
            # Write the log as a CSV file
            self.progress_log.to_csv(self.progress_log_name, index=False)


# =======================================================================================================================


def get_convolved_cube(reg, update=True, cnv_cubePath=None, edgetrim_width=5, paraNameRoot=None, paraDir=None,
                       multicore=True):
    if cnv_cubePath is None:
        root = "conv{0}Xbeam".format(int(np.rint(reg.cnv_factor)))
        reg.cnv_cubePath = "{0}_{1}.fits".format(os.path.splitext(reg.cubePath)[0], root)
    else:
        reg.cnv_cubePath = cnv_cubePath

    reg.cnv_para_paths = {}

    if update or (not os.path.isfile(reg.cnv_cubePath)):
        reg.ucube.convolve_cube(factor=reg.cnv_factor, savename=reg.cnv_cubePath, edgetrim_width=edgetrim_width)

    if paraNameRoot is None:
        paraNameRoot = "{}_conv{}Xbeam".format(reg.paraNameRoot, int(np.rint(reg.cnv_factor)))

    if paraDir is None:
        paraDir = reg.paraDir

    reg.ucube_cnv = UCube.UCubePlus(cubefile=reg.cnv_cubePath, paraNameRoot=paraNameRoot,
                                    paraDir=paraDir, cnv_factor=reg.cnv_factor, fittype=reg.fittype, n_cores=multicore)

    # MC: a mechanism is needed to make sure the convolved cube has the same resolution has the cnv_factor


def get_convolved_fits(reg, ncomp, update=True, **kwargs):
    '''
    update (bool) : call reg.get_convolved_cube even if reg has ucube_cnv attribute

    kwargs: passed to UCubePlus.fit_cube by reg.ucube_cnv.get_model_fit
    '''

    kwdf = dict(multicore=True)
    kwargs = {**kwdf, **kwargs}

    if not hasattr(reg, 'ucube_cnv'):
        reg.get_convolved_cube(update=True, multicore=kwargs['multicore'])
    else:
        reg.get_convolved_cube(update=update, multicore=kwargs['multicore'])

    try:
        reg.ucube_cnv.get_model_fit(ncomp, update=update, **kwargs)
    except StartFitError as e:
        msg = "Fits to convovled cube failed to start. " + e.__str__()
        pass


def get_fits(reg, ncomp, **kwargs):
    reg.ucube.get_model_fit(ncomp, **kwargs)


# =======================================================================================================================
# functions specific to 2-component fits


def master_2comp_fit(reg, snr_min=0.0, recover_wide=True, planemask=None, updateCnvFits=True, refit_bad_pix=True,
                     refit_marg=True, max_expand_iter=None, multicore=True):
    """
    Perform a two-component fit on the data cube within a Region object.

    This function applies a two-component fit to the spectral data, with optional adjustments for refining poor or
    marginal fits, recovering wide separation components, and expanding fits to surrounding regions based on
    signal-to-noise thresholds.

    Parameters
    ----------
    reg : Region object
        A Region object containing the data cube to be fitted.
    snr_min : float, optional
        Minimum peak signal-to-noise ratio required for fitting. Only regions with a peak SNR above this threshold
        are considered for fitting (default is 0.0). If set to 0.0, attempts to fit all pixels using guesses from
        pre-existing fits.
    recover_wide : bool, optional
        If True, attempts to recover spectral components with large velocity separation (default is True).
    planemask : ndarray, optional
        2D mask specifying which pixels to fit, allowing for targeted fitting within selected regions. If provided,
        this mask overrides any existing `planemask` within the Region object (default is None).
    updateCnvFits : bool, optional
        If True, performs fits on the convolved cube first, even if previous fits exist, to update or refine results
        (default is True).
    refit_bad_pix : bool, optional
        If True, refits pixels with poor fits to improve fit quality (default is True).
    refit_marg : bool, optional
        If True, refits pixels with marginally acceptable fits for better accuracy (default is True).
    max_expand_iter : int, bool, or None, optional
        Controls the maximum number of iterations for expanding the fitted regions:
        - If an integer, sets the exact number of expansion iterations.
        - If True or None, expands until no further expansion is possible, with limited expansion if `snr_min` > 3
          and larger expansion if `snr_min` is low.
        - If False, disables expansion entirely.
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing. If True, uses all available CPUs minus one. If an integer
        is provided, it specifies the exact number of CPU cores to use (default is True).

    Returns
    -------
    Region object
        Returns the modified `reg` object with updated fit results.

    Notes
    -----
    - The function first performs an initial two-component fit on the region, then optionally refits pixels with low SNR,
      wide separation, or marginal quality.
    - Further expansion of fits to surrounding pixels is possible if `max_expand_iter` is specified. When set to None or True,
      expansion proceeds with limited iterations if `snr_min` > 3, or more extensive iterations if `snr_min` is low.
    - If `match_footprint` is True within `fit_surroundings`, it attempts to align the fitted region’s footprints across
      components.
    - Final results are saved, with the best two-component fit retained based on log-likelihood thresholds.

    Examples
    --------
    >>> updated_reg = master_2comp_fit(region, snr_min=3.0, recover_wide=False, max_expand_iter=5, multicore=4)
    >>> # Performs a 2-component fit on the region with a minimum SNR of 3.0, without attempting wide recovery,
    >>> # and expands fits for 5 iterations.
    """

    iter_2comp_fit(reg, snr_min=snr_min, updateCnvFits=updateCnvFits, planemask=planemask, multicore=multicore)

    # assumes the user wants to recover a second component that is fainter than the primary
    recover_snr_min = 3.0
    if snr_min/3.0 < recover_snr_min:
        recover_snr_min = snr_min/3.0

    if refit_bad_pix:
        refit_bad_2comp(reg, snr_min=recover_snr_min, lnk_thresh=-5, multicore=multicore)

    if recover_wide:
        refit_2comp_wide(reg, snr_min=recover_snr_min, multicore=multicore)

    if refit_marg:
        refit_marginal(reg, ncomp=1, lnk_thresh=10, holes_only=False, multicore=multicore,
                       method='best_neighbour')
        refit_marginal(reg, ncomp=2, lnk_thresh=10, holes_only=False, multicore=multicore,
                       method='best_neighbour')

    if max_expand_iter != False:
        if max_expand_iter == True:
            max_expand_iter = None
        # perform expand fits
        fit_surroundings(reg, ncomps=[1, 2], snr_min=snr_min, max_iter=max_expand_iter, fill_mask=None,
                         multicore=multicore, match_footprint=True)
        if refit_bad_pix:
            # have another pass of quality assurance
            refit_bad_2comp(reg, snr_min=recover_snr_min, lnk_thresh=-5, multicore=multicore)

    save_best_2comp_fit(reg, multicore=multicore, lnk21_thres=5, lnk10_thres=5)

    return reg



def iter_2comp_fit(reg, snr_min=3.0, updateCnvFits=True, planemask=None, multicore=True, use_cnv_lnk=True,
                   save_para=True, lnk_cnv_thres=10):
    """
    Perform a two-component fit iterantively through two steps. The first step fits the convovle cube use moment-based guesses
    The second setp fits fits the cube at its native spatial resolution, using the results from the first iteration for guesses.

    Parameters
    ----------
    reg : Region object
        A Region object with the cube to be fitted
    snr_min : float, optional
        Minimum signal-to-noise ratio required for attempting fits (default is 3.0).
    updateCnvFits : bool, optional
        If True, fit the covlved cube first, even if it has been fitted before (default is True).
    planemask : ndarray, optional
        Mask specifying which spatial pixels to fit (default is None).
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
        If an integer is provided, it specifies the number of CPU cores to use.
    use_cnv_lnk : bool, optional
        If True,
    save_para : bool, optional
        If True, saves the fited parameters as . fits files after each iteration (default is True).

    Returns
    -------

    """

    multicore = validate_n_cores(multicore)
    logger.debug(f'Using {multicore} cores.')

    ncomp = [1, 2]  # ensure this is a two component fitting method

    # convolve the cube and fit it
    proc_name = f'iter conv fits'
    reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)
    reg.get_convolved_fits(ncomp, update=updateCnvFits, snr_min=snr_min, multicore=multicore) # actual fitting

    nc = 1
    pcube_cnv = reg.ucube_cnv.pcubes[str(nc)]
    reg.log_progress(process_name=proc_name, mark_start=False, n_attempted='N/A', n_success='N/A', finished=True)

    # use the result from the convolved cube as guesses for the full resolution fits
    for nc in ncomp:
        proc_name = f'iter {nc} comp fit'
        reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)

        pcube_cnv = reg.ucube_cnv.pcubes[str(nc)]
        para_cnv = np.append(pcube_cnv.parcube, pcube_cnv.errcube, axis=0)

        # establish where the new guesses should be extrapolated to
        mask = np.any(np.isfinite(para_cnv),axis=0)
        mask = remove_small_holes(mask)
        para_cnv[para_cnv == 0] = np.nan

        if use_cnv_lnk:
            # clean up the fits with lnk maps
            if nc == 2:
                lnk = reg.ucube_cnv.get_AICc_likelihood(2, 1)
            elif nc == 1:
                lnk = reg.ucube_cnv.get_AICc_likelihood(1, 0)

            # determine where the fits are bad based on local threshold, above the lnk_thresh global cutoff
            bad = get_local_bad(lnkmap=lnk, lnk_thresh=lnk_cnv_thres, block_size=15, offset=0, bad_size_max=225)
            bad = np.logical_or(bad, ~(lnk > lnk_cnv_thres))
            para_cnv[:, bad] = np.nan
            clean_map = False
        else:
            clean_map = True

        guesses = gss_rf.guess_from_cnvpara(para_cnv, reg.ucube_cnv.cube.header, reg.ucube.cube.header,
                                            clean_map=clean_map, tau_thresh=1, mask=mask)

        n_pix = np.all(np.isfinite(guesses), axis=0).sum()

        # update is set to True to save the fits
        kwargs = {'update': True, 'guesses': guesses, 'snr_min': snr_min, 'multicore': multicore}
        if planemask is not None:
            kwargs['maskmap'] = planemask
            n_pix = np.sum(np.all(np.isfinite(guesses), axis=0) & planemask)

        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=n_pix, n_success='N/A')

        reg.ucube.get_model_fit([nc], **kwargs)

        if save_para:
            save_updated_paramaps(reg.ucube, ncomps=[nc])

        n_good = reg.ucube.pcubes[str(nc)].has_fit.sum()

        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=None, n_success=n_good, finished=True)



def refit_bad_2comp(reg, snr_min=3, lnk_thresh=-5, multicore=True, save_para=True, method='best_neighbour', local_bad=True):

    """
    Refit pixels with poor 2-component fits, as determined by the log-likelihood of 2- and 1- compoent fits, using
    specified models

    Parameters
    ----------
    reg : Region object
        A Region object with the cube to be fitted
    snr_min : float, optional
        Minimum signal-to-noise ratio required for refitting (default is 3).
    lnk_thresh : float, optional
        Log-likelihood threshold used to identify pixels with pad fits (default is -5).
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
        If an integer is provided, it specifies the number of CPU cores to use.
    save_para : bool, optional
        If True, saves the fit parameters as .fits files after refitting (default is True).
    method : str, optional
        The method used for refitting bad pixels (default is 'best_neighbour').
    local_bad : bool
        If True, include bad pixels found from local-thresholding

    Returns
    -------

    """
    logger.info("Begin re-fitting bad 2-comp pixels")
    multicore = validate_n_cores(multicore)
    logger.debug(f'Using {multicore} cores.')

    proc_name = 'refit_bad_2comp'
    reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)
    ucube = reg.ucube

    from astropy.convolution import Gaussian2DKernel, convolve

    lnk10, lnk20, lnk21 = reg.ucube.get_all_lnk_maps(ncomp_max=2, rest_model_mask=False, multicore=multicore)

    # where the fits are poor
    mask = np.logical_or(lnk21 < lnk_thresh, lnk20 < 5)
    mask = np.logical_and(mask, lnk10 > 5)
    mask = np.logical_and(mask, np.isfinite(lnk10))

    if local_bad:
        bad_thresh = 10 # threshold value ot look for realtively bad pixels
        if lnk_thresh >=bad_thresh:
            bad_thresh = lnk_thresh
        bad = get_local_bad(lnkmap=lnk21, lnk_thresh=bad_thresh, block_size=15, offset=0, bad_size_max=225)
        mask = np.logical_or(mask, bad)

    mask_size = np.sum(mask)
    if mask_size > 0:
        logger.info("Attempting to refit over {} pixels to recover bad 2-comp. fits".format(mask_size))
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=mask_size, n_success='in progress')
    else:
        logger.info("No pixel was used in attempt to recover bad 2-comp. fits")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=0, finished=True)
        return

    guesses, mask = get_refit_guesses(ucube, mask, ncomp=2, method='best_neighbour', refmap=lnk20)
    # re-fit and save the updated model
    good_mask = replace_bad_pix(ucube, mask, snr_min, guesses, ncomp=2, simpfit=True,
                                multicore=multicore, return_good_mask=True)

    if save_para:
        save_updated_paramaps(reg.ucube, ncomps=[2, 1])

    reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=None, n_success=good_mask.sum(), finished=True)



def refit_marginal(reg, ncomp, lnk_thresh=5, holes_only=False, multicore=True, save_para=True, method='best_neighbour',
                   refit_only=True, **kwargs_marg):
    """
    Refit pixels with fits that appears marginally okay, as deterined by the specified log-likelihood threshold provided

    Parameters
    ----------
    reg : Region object
        A Region object with the cube to be fitted
    ncomp : int
        Number of components of the model to be refitted
    lnk_thresh : float, optional
        Log-likelihood threshold used to determine the pixels to refit (default is 5).
    holes_only : bool, optional
        If True, only refits pixels surrounded by good fits (default is False).
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
        If an integer is provided, it specifies the number of CPU cores to use.
    save_para : bool, optional
        If True, saves the fit parameters as .fits files after refitting (default is True).
    method : str, optional
        The method used for refitting marginal pixels (default is 'best_neighbour').
    **kwargs_marg : dict, optional
        Additional keyword arguments for fine-tuning the marginal refitting process.

    Returns
    -------

    """

    ucube = reg.ucube
    multicore = validate_n_cores(multicore)
    logger.info(f"Begin re-fitting marginal pixels using {multicore} cores")

    proc_name = f'refit_marginal_{ncomp}_comp'
    reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)
    ucube = reg.ucube

    lnk_maps = reg.ucube.get_all_lnk_maps(ncomp_max=ncomp, rest_model_mask=False, multicore=multicore)
    if ncomp == 1:
        lnkmap = lnk_maps #lnk10
        refmap = lnkmap
    elif ncomp == 2:
        lnkmap = lnk_maps[2] #lnk21 for thresholding
        refmap = lnk_maps[1] #lnk20 for best neighbour (in case lnk21 is high simply the one component fit is poor)

    mask = get_marginal_pix(lnkmap, lnk_thresh=lnk_thresh, holes_only=holes_only, **kwargs_marg)
    if mask.sum() < 1:
        logger.info(f"No pixel was used in attempt to recover marginal {ncomp}-comp. fits")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=0, finished=True)
        return

    guesses, mask = get_refit_guesses(ucube, mask=mask, ncomp=ncomp, method=method, refmap=lnkmap)

    if refit_only:
        # ensure the mask doesn't extend beyond the original fit and has guesses
        mask = np.logical_and(mask, np.isfinite(guesses).all(axis=0))
        mask = np.logical_and(mask, reg.ucube.pcubes[str(ncomp)].has_fit)

    mask_size = np.sum(mask)
    if mask_size > 0:
        logger.info(f"Attempting to refit over {mask_size} pixels to recover marginal {ncomp}-comp. fits")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=mask_size, n_success='in progress')
    else:
        logger.info(f"No pixel was used in attempt to recover marginal {ncomp}-comp. fits")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=0, finished=True)
        return

    # re-fit and save the updated model
    snr_min=0
    good_mask = replace_bad_pix(ucube, mask, snr_min, guesses, ncomp=ncomp, simpfit=True, multicore=multicore, return_good_mask=True)

    if save_para:
        save_updated_paramaps(reg.ucube, ncomps=[2, 1])

    reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=None, n_success=good_mask.sum(), finished=True)



def refit_swap_2comp(reg, snr_min=3):
    """
    Refit the cube by using the previous fit result as guesses, but with the front and rear components switched.

    Parameters
    ----------
    reg : Region object
        The Region object containing the cube to be refitted
    snr_min : float, optional
        Minimum signal-to-noise ratio required for a pixel to be refitted (default is 3).

    Returns
    -------
    None
    """

    ncomp = [1, 2]

    # load the fitted parameters
    for nc in ncomp:
        if not str(nc) in reg.ucube.pcubes:
            # no need to worry about wide seperation as they likley don't overlap in velocity space
            reg.ucube.load_model_fit(reg.ucube.paraPaths[str(nc)], nc, reg.fittype)

    # refit only over where two component models are already determined to be better
    # note: this may miss a few fits where the swapped two-comp may return a better result?
    lnk21 = reg.ucube.get_AICc_likelihood(2, 1)
    mask = lnk21 > 5

    gc.collect()

    # swap the parameter of the two slabs and use it as the initial guesses
    guesses = copy(reg.ucube.pcubes['2'].parcube)

    for i in range(4):
        guesses[i], guesses[i + 4] = guesses[i + 4], guesses[i]

    ucube_new = UCube.UltraCube(reg.ucube.cubefile, fittype=reg.fittype)
    ucube_new.fit_cube(ncomp=[2], maskmap=mask, snr_min=snr_min, guesses=guesses)

    gc.collect()

    # do a model comparison between the new two component fit verses the original one
    lnk_NvsO = UCube.calc_AICc_likelihood(ucube_new, 2, 2, ucube_B=reg.ucube)

    gc.collect()

    # adopt the better fit parameters into the final map
    good_mask = lnk_NvsO > 0
    # replace the values
    replace_para(reg.ucube.pcubes['2'], ucube_new.pcubes['2'], good_mask)

    # re-fit and save the updated model
    save_updated_paramaps(reg.ucube, ncomps=[2, 1])


def refit_2comp_wide(reg, snr_min=3, method='residual', planemask=None, multicore=True, save_para=True):
    """
    Refit pixels to recover compoents with wide velocity separation for 2-component models

    Parameters
    ----------
    reg : Region object
        A Region object with the cube to be fitted
    snr_min : float, optional
        Minimum signal-to-noise ratio required for refitting (default is 3).
    method : {'residual', 'moments'}, optional
        Method used to recover the wide component. 'residual' uses the residual cube to recover the wide component,
        while 'moments' uses moment-based guesses (default is 'residual').
    planemask : ndarray, optional
        Mask specifying which spatial pixels to fit (default is None). If not provided, only pixels where the
        two-component fit is worse than the one-component fit are used.
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
        If an integer is provided, it specifies the number of CPU cores to use.
    save_para : bool, optional
        If True, saves the fit parameters as .fits files after refitting (default is True).

    Returns
    -------
    None

    """
    logger.info("Begin wide component recovery")
    multicore = validate_n_cores(multicore)
    logger.debug(f'Using {multicore} cores.')

    proc_name = 'refit_2comp_wide'
    reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)

    ncomp = [1, 2]

    # load the fitted parameters
    for nc in ncomp:
        if not str(nc) in reg.ucube.pcubes:
            if not str(nc) in reg.ucube.paraPaths:
                reg.ucube.paraPaths[str(nc)] = '{}/{}_{}vcomp.fits'.format(reg.ucube.paraDir, reg.ucube.paraNameRoot,
                                                                           nc)

            if nc == 2 and not ('2_noWideDelV' in reg.ucube.paraPaths):
                reg.ucube.load_model_fit(reg.ucube.paraPaths[str(nc)], nc, reg.fittype, multicore=multicore)
                reg.ucube.pcubes['2_noWideDelV'] = \
                    "{}_noWideDelV".format(os.path.splitext(reg.ucube.paraPaths[str(nc)])[0])
            else:
                reg.ucube.load_model_fit(reg.ucube.paraPaths[str(nc)], nc, reg.fittype, multicore=multicore)

    if method == 'residual':
        logger.debug("recovering second component from residual")
        # fit over where one-component was a better fit in the last iteration (since we are only interested in recovering
        # a second component that is found in wide separation)
        lnk10, lnk20, lnk21 = reg.ucube.get_all_lnk_maps(ncomp_max=2, rest_model_mask=False)

        if planemask is None:
            mask10 = lnk10 > 5
            mask = np.logical_and(lnk21 < -5, lnk10 > 5)
        else:
            mask10 = planemask
            mask = planemask

        # use the one component fit and the refined 1-component guess for the residual to perform the two components fit
        c1_guess = copy(reg.ucube.pcubes['1'].parcube)
        c1_guess = gss_rf.refine_each_comp(c1_guess)

        try:
            wide_comp_guess = get_2comp_wide_guesses(reg, window_hwidth=3.5, snr_min=snr_min, savefit=True,
                                                     planemask=mask)
        except SNRMaskError as e:
            msg = e.__str__() + " No second component recovered from the residual cube."
            logger.warning(msg)
            reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=0, finished=True)
            return
        except StartFitError as e:
            logger.warning(e.__str__())
            return
        except ValueError as e:
            msg = e.__str__() + "Convolved residual cube is not fitted. No recovery is done for the wide component."
            logger.warning(msg)
            reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=0, finished=True)
            return
        # reduce the linewidth guess to avoid overestimation
        wide_comp_guess[:, ~mask] = np.nan

        final_guess = np.append(c1_guess, wide_comp_guess, axis=0)

        mask = np.logical_and(mask, np.all(np.isfinite(final_guess), axis=0))

        # further refine to correct for the typical error estimate for nh3 residual guesses
        final_guess = gss_rf.refine_2c_guess(final_guess)
        simpfit = True

    elif method == 'moments':
        # this method could be computationally expensive if the mask contains a larger number of pixels
        final_guess = mmg.mom_guess_wide_sep(reg.ucube.cube, planemask=mask)
        mask = np.logical_and(mask, np.all(np.isfinite(final_guess), axis=0))
        simpfit = True

    else:
        raise Exception("the following method specified is invalid: {}".format(method))

    mask_size = np.sum(mask)
    if mask_size > 0:
        logger.info("Attempting wide recovery over {} pixels".format(mask_size))
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=mask_size, n_success='in progress')
    else:
        logger.info("No pixel was used in attempt to recover wide component")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=None, finished=True)
        return

    good_mask = replace_bad_pix(reg.ucube, mask, snr_min, final_guess, ncomp=2, simpfit=simpfit,
                             multicore=multicore, return_good_mask=True)

    if save_para:
        save_updated_paramaps(reg.ucube, ncomps=[2, 1])

    reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=None, n_success=good_mask.sum(), finished=True)


def expand_fits(reg, ncomp, lnk_thresh=5, max_iter=5, r_expand=1, fill_mask=None, lnkmap=None, multicore=True, snr_min=0,
               save_para=True, return_niter=False):
    """
    Expand fits in a region by incrementally fitting pixels beyond a defined log-likelihood threshold boundary.

    This function iteratively expands fitted regions by attempting to fit pixels beyond a boundary set by the
    relative log-likelihood threshold.

    Parameters
    ----------
    reg : Region object
        The region object containing the data cube and the fitting model.
    ncomp : int
        Number of components in the model to fit in each pixel.
    lnk_thresh : float, optional
        Relative log-likelihood threshold used to define the boundary for expansion. Pixels with log-likelihood
        values above this threshold are considered for expansion (default is 5).
    max_iter : int, optional
        Maximum number of expansion iterations to perform. Each iteration attempts to fit the next layer of pixels
        beyond the current boundary (default is 5).
    r_expand : int, optional
        Number of pixels to expand in each direction during each iteration (default is 1).
    fill_mask : ndarray, optional
        Boolean array that defines the outer boundary for expansion. Expansion is limited to pixels within this mask
        (default is None, which implies no additional mask constraint).
    lnkmap : ndarray, optional
        A precomputed 2D array (i.e., map) of relative log-likelihood values. Supplying this map saves computation time by avoiding
        recalculating these values. If None, the function will compute the map as needed (default is None).
    multicore : bool or int, optional
        Whether to use multiple cores for parallel processing. If True, automatically selects available cores;
        if an integer is provided, it specifies the number of cores (default is True).
    snr_min : float, optional
        Minimum signal-to-noise ratio threshold for successful fits. Fits below this threshold are not considered "good"
        (default is 0).
    save_para : bool, optional
        If True, saves updated parameter maps after fitting. Useful for keeping track of intermediate fit results
        (default is True).
    return_niter : bool, optional
        If True, returns both the total number of successfully fitted pixels and the number of iterations performed
        (default is False).

    Returns
    -------
    int or tuple of (int, int)
        If `return_niter` is False, returns the total number of successfully fitted pixels across all iterations.
        If `return_niter` is True, returns a tuple containing the total number of successfully fitted pixels and
        the number of iterations completed.

    Examples
    --------
    >>> n_good_total = expand_fits(reg, ncomp=2, lnk_thresh=10, max_iter=3, r_expand=1, multicore=4)
    >>> print(f"Total good fits: {n_good_total}")

    """

    logger.debug("Begin fitting expanded footprints")
    multicore = validate_n_cores(multicore)
    logger.debug(f'Using {multicore} cores.')

    proc_name = f'expand_fit_{ncomp}comp'
    reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)

    if lnkmap is None:
        # note: assume ncomp - 1 has been fitted already
        lnkmap = reg.ucube.get_AICc_likelihood(ncomp, ncomp - 1)

    if fill_mask is None:
        fill_mask = np.ones(lnkmap.shape, dtype=bool)

    def expand_mask(lnkmap, lnk_thresh, fill_mask, r=1):
        # expand the mask and return the outer most layer
        mask = lnkmap > lnk_thresh
        mask_xor = np.logical_xor(mask, binary_dilation(mask, footprint=disk(r)))
        return mask_xor & fill_mask

    def fit(reg, lnkmap, mask, ncomp, multicore, r=2, snr_min=0):
        # fit the pixels in the mask using best_neighbour guesses, returen number of good fits
        guesses, mask = get_refit_guesses(reg.ucube, mask, ncomp=ncomp, method='best_neighbour', refmap=lnkmap,
                                          structure=neighbours.disk_neighbour(r))
        return replace_bad_pix(reg.ucube, mask, snr_min, guesses, ncomp=ncomp, simpfit=True, multicore=multicore,
                                  return_good_mask=True)

    # locations to fit
    mask_fit = expand_mask(lnkmap=lnkmap, lnk_thresh=lnk_thresh, fill_mask=fill_mask, r=r_expand)

    # location of the attempted fits
    mask_attempted = mask_fit

    if mask_fit.sum() < 1:
        logger.info(f"No pixel qualified for new fits for fitting expansion.")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=0, n_success=0,
                         finished=True)
        if return_niter:
            return 0, 0, 0
        else:
            return 0, 0

    good_mask = fit(reg, lnkmap=lnkmap, mask=mask_fit, ncomp=ncomp, multicore=multicore, r=r_expand+1)
    bad_mask = np.logical_and(mask_fit, ~good_mask)
    bad_twice = np.zeros(bad_mask.shape, dtype=bool)
    n_good = good_mask.sum()
    n_attempt = mask_fit.sum()
    n_good_total = n_good

    i = 0
    # iterate as long as there are good pixels to fit or if max_iter is reached
    while (i < max_iter) and (n_good > 0):
        i += 1
        lnkmap[mask_fit] = reg.ucube.get_AICc_likelihood(ncomp, ncomp - 1, planemask=mask_fit)
        mask_fit = expand_mask(lnkmap=lnkmap, lnk_thresh=lnk_thresh, fill_mask=fill_mask, r=r_expand)

        # only attempt to fit pixels that haven't been fitted, or only had one failed attempt
        mask_fit = np.logical_and(mask_fit, ~mask_attempted)
        mask_fit = np.logical_or(mask_fit, bad_mask & ~bad_twice)

        if mask_fit.sum() == 0:
            break
        # update the number of attempted and good fits
        good_mask = fit(reg, lnkmap=lnkmap, mask=mask_fit, ncomp=ncomp, multicore=True, r=r_expand+1)
        bad_mask = np.logical_and(mask_fit, ~good_mask)
        bad_twice = np.logical_or(bad_twice, bad_mask)

        # update where fits were attempted
        mask_attempted = np.logical_or(mask_attempted, mask_fit)

        n_good = good_mask.sum()
        n_attempt = mask_fit.sum()
        n_good_total += n_good

        logger.debug(f"Number of good fits at iteration {i} of expand fit: {n_good}/{n_attempt}")
        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=mask_attempted.sum(), n_success=n_good_total,
                         finished=False)

    if save_para:
        save_updated_paramaps(reg.ucube, ncomps=[2, 1])

    n_attempt_total = mask_attempted.sum()
    logger.info(f"Total good fits from expand fits for {ncomp} comp: {n_good_total}/{n_attempt_total}")

    reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=n_attempt_total, n_success=n_good_total, finished=True)

    if return_niter:
        return n_good_total, n_attempt_total, i
    else:
        return n_good_total, n_attempt_total


def fit_surroundings(reg, ncomps=[1, 2], snr_min=3, max_iter=None, save_para=True, fill_mask=None, multicore=True,
                     match_footprint=True):
    """
    Expand fits around a region based on model log-likelihood thresholds and target signal-to-noise depth.

    This function iteratively expands fitted regions by incrementally adjusting thresholds, expanding boundaries
    based on signal-to-noise ratios and log-likelihood values, and aligning footprints across multiple components.

    Parameters
    ----------
    reg : Region object
        The region containing data and models to be fitted and expanded.
    ncomps : list of int, optional
        Specifies the number of components for each model to fit in the region. Each component count in the list
        is processed in ascending order, expanding the fit boundaries accordingly (default is [1, 2]).
    snr_min : float, optional
        Minimum signal-to-noise ratio threshold, used to dynamically adjust `max_iter` if not explicitly set.
        Higher values restrict expansions to higher-SNR regions only. If `snr_min` is less than 3, expansion may
        extend to lower SNR areas (default is 3).
    max_iter : int, optional
        Maximum number of iterations for expanding the fitted regions. If `None`, defaults based on `snr_min`:
        a high `snr_min` will limit expansion, while a low `snr_min` allows for more iterations (default is None).
    save_para : bool, optional
        If True, saves the fitted parameter maps after each iteration (default is True).
    fill_mask : ndarray, optional
        Boolean array defining the boundary for expansion; only pixels within this mask are considered for expansion.
        Defaults to the region’s default mask if None.
    multicore : bool or int, optional
        Enables or sets the number of cores for parallel processing. If True, uses available cores automatically;
        if an integer, specifies the number of cores explicitly (default is True).
    match_footprint : bool, optional
        If True, aligns the fitted regions' footprints for different models by updating the `fill_mask` after each fit,
        ensuring consistent regions across component models. This parameter affects `max_iter`, setting it to `max_iter_fill`
        (default is True).

    Returns
    -------
    None

    Notes
    -----
    - Expansion proceeds by adjusting `lnk_thresh` for robustness: starting with a high threshold (10) for reliable fits,
      followed by a standard threshold (5) and optionally lower thresholds if `snr_min` is below 3.
    - If `match_footprint` is True, the function will use `max_iter_fill` to expand each model component’s footprint
      and attempt to match it with the previous component’s footprint.
    - The `max_iter` parameter dynamically controls the extent of expansion based on `snr_min`, promoting efficient
      use of resources while ensuring sufficient expansion depth.

    Examples
    --------
    >>> fit_surroundings(region, ncomps=[1, 2], snr_min=0, max_iter=10, multicore=4)
    >>> # Expands and fits regions with up to 2 components, filling as much of the map as the maximum iterations allow.
    """

    logger.info("Begin expand fitting the surroundings")
    multicore = validate_n_cores(multicore)
    logger.debug(f'Using {multicore} cores.')

    # whether or not to expand regions with expected lnk < 5
    low_signal = True
    if snr_min > 0:
        low_signal = False

    # the maximum number of iterations for ncomp beyond the smallest one if match_footprint is True
    max_iter_fill = 100

    # set default iteration number
    if max_iter is None:
        if snr_min >= 3:
            max_iter = 1
        else:
            max_iter = max_iter_fill
    elif max_iter > max_iter_fill:
        # match max_iter_fill to the user provided value if the latter is larger
        max_iter_fill = max_iter

    if fill_mask is None:
        fill_mask = np.any(reg.ucube.cube.mask.include(), axis=0)

    # sort the ncomp list in asending order so the lowest compoent gets processed first
    ncomps.sort()

    for i, ncomp in enumerate(ncomps):
        proc_name = f'expand_fit_{ncomp}comp'
        reg.log_progress(process_name=proc_name, mark_start=True, cores=multicore)

        m_iter = max_iter
        if match_footprint:
            if i > 0:
                # use max_iter_fill in attempt to match the footprints
                m_iter = max_iter_fill

                # update fill_mask if matching refittin footprint (up to max_iter) is desired
                ncomp_previous = str(ncomps[i-1])
                if i > 1:
                    fill_mask = np.logical_or(fill_mask, reg.ucube.pcubes[ncomp_previous].has_fit)
                else:
                    fill_mask = reg.ucube.pcubes[ncomp_previous].has_fit

        n_good_total = 0
        n_attempt_total = 0
        kwargs = dict(ncomp=ncomp, save_para=save_para, fill_mask=fill_mask, multicore=multicore, return_niter=True, snr_min=snr_min)

        # start expanding with a lnk_thresh of 10 for extra robustness
        lnk_thresh = 10
        # refit marginal first to ensure quality
        n_good, n_attempt, i = expand_fits(reg, lnk_thresh=lnk_thresh, max_iter=m_iter, r_expand=1, **kwargs)
        n_good_total += n_good
        n_attempt_total += n_attempt
        m_iter -= i

        # expand to a threshold of 5 for standard robustness
        if m_iter > 0 and n_good>0:
            lnk_thresh = 5
            n_good, n_attempt, i =  expand_fits(reg, lnk_thresh=lnk_thresh, max_iter=max_iter, r_expand=1, **kwargs)
            n_good_total += n_good
            n_attempt_total += n_attempt
            m_iter -= i

        # expand to lower thresholds with larger expansions
        if low_signal:

            if m_iter > 0 and n_good > 0:
                n_good, n_attempt, i =  expand_fits(reg, lnk_thresh=0, max_iter=max_iter, r_expand=2, **kwargs)
                n_good_total += n_good
                n_attempt_total += n_attempt
                m_iter -= i

            if m_iter > 0 and n_good > 0:
                n_good, n_attempt, i =  expand_fits(reg, lnk_thresh=-20, max_iter=max_iter, r_expand=3, **kwargs)
                n_good_total += n_good
                n_attempt_total += n_attempt
                m_iter -= i

        reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=n_attempt_total, n_success=n_good_total,
                         finished=True)



def replace_bad_pix(ucube, mask, snr_min, guesses, ncomp=2, lnk21=None, simpfit=True, multicore=True, return_good_mask=False):
    """
    Refit pixels marked by the mask as "bad" and adopt the new model if it is determined to be better.

    Parameters
    ----------
    ucube : UltraCube object
        The UltraCube object containing the spectral data to be refitted.
    mask : ndarray
        A boolean mask indicating the pixels to be refitted.
    snr_min : float
        Minimum signal-to-noise ratio required for refitting.
    guesses : ndarray
        Initial guesses for the refitting process.
    lnk21 : None, optional
        Deprecated. This argument is retained for backward compatibility but will be removed in a future update.
    simpfit : bool, optional
        If True, use simplified fitting (default is True).
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
        If an integer is provided, it specifies the number of CPU cores to use.
    return_good_mask : bool
        If True, return the number of pixels with good fits

    Returns
    -------
    None

    """

    if lnk21 is not None:
        message = "[WARNING] The lnk21 argument will be deprecated in a future update."
        warnings.warn(message, DeprecationWarning, stacklevel=2)

    if np.sum(mask) >= 1:
        ucube_new = UCube.UltraCube(ucube.cubefile, fittype=ucube.fittype)
        # fit using simpfit (and take the provided guesses as they are)
        try:
            ucube_new.fit_cube(ncomp=[ncomp], simpfit=simpfit, maskmap=mask, snr_min=snr_min,
                               guesses=guesses, multicore=multicore)
        except SNRMaskError:
            logger.info("No valid pixel to refit with snr_min={}."
                        " Please consider trying a lower snr_min value.".format(snr_min))
            if return_good_mask:
                return np.zeros((1,1), dtype=bool)
            else:
                return

        # do a model comparison between the new two component fit verses the original one
        lnk_NvsO = UCube.calc_AICc_likelihood(ucube_new, ncomp, ncomp, ucube_B=ucube)

        good_mask = np.logical_and(lnk_NvsO > 0, mask)

        logger.info("Replacing {} bad pixels with better fits".format(good_mask.sum()))
        # replace the values
        replace_para(ucube.pcubes[str(ncomp)], ucube_new.pcubes[str(ncomp)], good_mask, multicore=multicore)
        replace_rss(ucube, ucube_new, ncomp=ncomp, mask=good_mask)
    else:
        logger.debug("not enough pixels to refit, no-refit is done")
        np.zeros((1,1), dtype=bool)

    if return_good_mask:
        return good_mask



def replace_rss(ucube, ucube_ref, ncomp, mask):
    compID = str(ncomp)

    attrs = ['rss_maps', 'NSamp_maps', 'AICc_maps']
    for attr in attrs:
        if attr != 'AICc_maps':
            try:
                getattr(ucube, attr)[compID][mask] = copy(getattr(ucube_ref, attr)[compID][mask])
            except KeyError:
                logger.debug("{} does not have the following key: {}".format(attr, compID))
        else:
            ucube.get_AICc(ncomp=ncomp, update=True, planemask=mask)

def get_refit_guesses(ucube, mask, ncomp, method='best_neighbour', refmap=None, structure=None):
    #get refit guesses from the surrounding pixels

    if structure is None:
        structure = neighbours.square_neighbour(1)

    guesses = copy(ucube.pcubes[str(ncomp)].parcube)
    guesses[guesses == 0] = np.nan
    # remove the bad pixels from the fitted parameters
    guesses[:, mask] = np.nan

    if method == 'best_neighbour':
        if refmap is None:
            raise ValueError("refmap must be provided for the best_neighbour method.")
        if not isinstance(refmap, np.ndarray):
            raise TypeError(f"{type(refmap)} is the incorrect type for refmap.")
        # use the nearest neighbour with the highest lnk20 value for guesses
        # neighbours.square_neighbour(1) gives the 8 closest neighbours
        maxref_coords = neighbours.maxref_neighbor_coords(mask=mask, ref=refmap, fill_coord=(0, 0),
                                                          structure=structure)
        ys, xs = zip(*maxref_coords)
        guesses[:, mask] = guesses[:, ys, xs]
        mask = np.logical_and(mask, np.all(np.isfinite(guesses), axis=0))

    elif method == 'convolved':
        # use astropy convolution to interpolate guesses (we assume bad fits are usually well surrounded by good fits)
        kernel = Gaussian2DKernel(2.5 / 2.355)
        for i, gmap in enumerate(guesses):
            gmap[mask] = np.nan
            guesses[i] = convolve(gmap, kernel, boundary='extend')

    return guesses, mask


def get_marginal_pix(lnkmap, lnk_thresh=5, holes_only=False, smallest_struct_size=9):
    """
    Return pixels at the edge of structures with values greater than `lnk_thresh`, or pixels less than `lnk_thresh`
    enclosed within the structures.

    Parameters
    ----------
    lnkmap : ndarray
        The relative log-likelihood map.
    lnk_thresh : float, optional
        The relative log-likelihood threshold for identifying structures (default is 5).
    holes_only : bool, optional
        If True, return only the holes surrounded by pixels with values greater than `lnk_thresh` (default is False).
    smallest_struct_size : int, optional
        The minimum size of connected pixels to be considered a good reference structure (default is 9).

    Returns
    -------
    ndarray
        A boolean mask indicating the marginal pixels, either at the edges of structures or within enclosed regions.
    """

    mask = remove_small_objects(lnkmap > lnk_thresh, smallest_struct_size)
    mask_nosml = remove_small_holes(mask)

    if holes_only:
        # returns holes surrounded by pixels with lnk > lnk_thresh
        return np.logical_xor(mask_nosml, mask)

    else:
        mask_nosml = binary_dilation(mask_nosml)
        return np.logical_xor(mask_nosml, mask)


def get_local_bad(lnkmap, lnk_thresh=5, block_size=15, offset=0, bad_size_max=225):
    """
    Identify local pixels with significantly lower relative log-likelihood (lnk) values compared to their neighbors.

    Parameters
    ----------
    lnkmap : ndarray
        The relative log-likelihood map
    lnk_thresh : float, optional
        Global threshold for the relative log-likelihood values. Only pixels with lnk values above this threshold are evaluated
        (default is 5).
    block_size : int, optional
        Size of the local region (in pixels) used to compute the threshold for each pixel. Must be an odd number
        (default is 15).
    offset : float, optional
        Constant subtracted from the computed local threshold to adjust sensitivity. Higher values make the threshold more lenient
        (default is 0).
    bad_size_max : int, optional
        Maximum area (in pixels) for isolated regions of bad pixels. Larger regions remain unaffected
        (default is 225).

    Returns
    -------
    mask : ndarray
        A boolean array of the same shape as `lnkmap`, where `True` indicates "bad" pixels.
    """
    local_thresh = threshold_local(lnkmap, block_size, offset=offset)
    binary_local = lnkmap > local_thresh
    good = binary_local & (lnkmap > lnk_thresh)
    # holes inside relative thresholds are the bad pixels
    mask = remove_small_holes(good, bad_size_max)
    mask = np.logical_xor(mask, good)
    return mask



def standard_2comp_fit(reg, planemask=None, snr_min=3):
    """
    Perform a two-component fit for the cube using default moment map guesses.

    Parameters
    ----------
    reg : Region
        The Region object containing the cube to be fitted.
    planemask : ndarray, optional
        Mask specifying which spatial pixels to fit (default is None). If provided, fitting is limited to the masked area.
    snr_min : float, optional
        Minimum signal-to-noise ratio required for attempting fits (default is 3).

    Returns
    -------
    None
    """
    proc_name = 'standard_2comp_fit'
    reg.log_progress(process_name=proc_name, mark_start=True)
    ncomp = [1, 2]

    # Only use the moment maps for the fits
    for nc in ncomp:
        # Update is set to True to save the fits
        kwargs = {'update': True, 'snr_min': snr_min}
        if planemask is not None:
            kwargs['maskmap'] = planemask
        reg.ucube.get_model_fit([nc], fittype=reg.fittype, **kwargs)

    reg.log_progress(process_name=proc_name, mark_start=False, n_attempted=None, n_success=None)


def save_updated_paramaps(ucube, ncomps):
    """
    Save the updated parameter maps for specified components.

    Parameters
    ----------
    ucube : UltraCube
        The UltraCube object containing the spectral data.
    ncomps : list of int
        List of the number of components of the model for which parameter maps will be saved.

    Returns
    -------
    None
    """
    for nc in ncomps:
        if str(nc) not in ucube.paraPaths:
            logger.error("The ucube does not have paraPath for '{}' components".format(nc))
        else:
            ucube.save_fit(ucube.paraPaths[str(nc)], nc)



def save_best_2comp_fit(reg, multicore=True, from_saved_para=False, lnk21_thres=5, lnk10_thres=5):
    """
    Save the best two-component fit results for the specified region.

    Parameters
    ----------
    reg : Region
        The Region object containing the cube and fit results.
    multicore : bool or int, optional
        Number of CPU cores to use for parallel processing (default is True, which uses all available CPUs minus 1).
        If an integer is provided, it specifies the number of CPU cores to use.
    from_saved_para : bool, optional
        If True, reload parameters from saved files instead of using existing results in memory (default is False).
    lnk21_thres : float, optional
        The log-relative-likelihood theshold to select the 2-component model over the 1-component model (default is 5)
    lnk10_thres : float, optional
        The log-relative-likelihood theshold to select the 1-component model over the noise model (default is 5)

    Returns
    -------
    None
    """

    ncomps = [1, 2]
    multicore = validate_n_cores(multicore)

    # ensure the latest results were saved
    save_updated_paramaps(reg.ucube, ncomps=ncomps)

    if from_saved_para:
        # create a new Region object to start fresh
        reg_final = Region(reg.cubePath, reg.paraNameRoot, reg.paraDir, fittype=reg.fittype, initialize_logging=False)

        # load files using paths from reg if they exist
        for nc in ncomps:
            if not str(nc) in reg.ucube.pcubes:
                reg_final.load_fits(ncomp=[nc])
            else:
                logger.debug("Loading model from: {}".format(reg.ucube.paraPaths[str(nc)]))
                reg_final.ucube.load_model_fit(filename=reg.ucube.paraPaths[str(nc)], ncomp=nc, multicore=multicore)
    else:
        reg_final = reg

    # make the two-component parameter maps with the best fit model
    pcube_final = reg_final.ucube.pcubes['2']
    kwargs = dict(multicore=multicore, lnk21_thres=lnk21_thres, lnk10_thres=lnk10_thres, return_lnks=True)
    parcube, errcube, lnk10, lnk20, lnk21 = reg_final.ucube.get_best_2c_parcube(**kwargs)
    pcube_final.parcube = parcube
    pcube_final.errcube = errcube

    # use the default file format to save the final results
    nc = 2
    if not str(nc) in reg_final.ucube.paraPaths:
        reg_final.ucube.paraPaths[str(nc)] = '{}/{}_{}vcomp.fits'.format(
            reg_final.ucube.paraDir, reg_final.ucube.paraNameRoot, nc
        )

    savename = "{}_final.fits".format(os.path.splitext(reg_final.ucube.paraPaths['2'])[0])
    notes = 'Model-selected best 1- or 2-comp fits parameters, based on lnk21'
    UCube.save_fit(pcube_final, savename=savename, ncomp=2, header_note=notes)

    hdr2D = reg.ucube.make_header2D()
    paraDir = reg_final.ucube.paraDir
    paraRoot = reg_final.ucube.paraNameRoot

    def make_lnk_header(ref_header, root):

        hdr_new = ref_header.copy()

        comp_a, comp_b = root[0], root[1]
        if comp_b == '0':
            comp_b = 'noise'
        else:
            comp_b = f'{comp_b} comp fit'

        hdr_new.set(keyword='NOTES', value=f"Relative log-likelihood of {comp_a} comp fit vs {comp_b}",
                     comment=None, before='DATE')
        hdr_new.set(keyword='BUNIT', value='unitless', comment=None, before=1)

        return hdr_new


    def make_save_name(paraRoot, paraDir, key):
        """
        Generate the filename for saving parameter maps, based on the provided root and directory.

        Parameters
        ----------
        paraRoot : str
            Root name of the parameter map files.
        paraDir : str
            Directory to save the parameter map files.
        key : str
            Key to differentiate the saved parameter map files.

        Returns
        -------
        str
            Full path for saving the parameter map file.
        """
        if "parameters" in paraRoot:
            return "{}/{}.fits".format(paraDir, paraRoot.replace("parameters", key))
        elif "parameter" in paraRoot:
            return "{}/{}.fits".format(paraDir, paraRoot.replace("parameter", key))
        elif "para" in paraRoot:
            return "{}/{}.fits".format(paraDir, paraRoot.replace("para", key))
        else:
            return "{}/{}_{}.fits".format(paraDir, paraRoot, key)


    # save the lnk21 map
    savename = make_save_name(paraRoot, paraDir, "lnk21")
    hdr_save = make_lnk_header(ref_header=hdr2D, root='21')
    save_map(lnk21, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))

    # save the lnk10 map
    savename = make_save_name(paraRoot, paraDir, "lnk10")
    hdr_save = make_lnk_header(ref_header=hdr2D, root='10')
    save_map(lnk10, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))

    # save the lnk20 map for reference:
    savename = make_save_name(paraRoot, paraDir, "lnk20")
    hdr_save = make_lnk_header(ref_header=hdr2D, root='20')
    save_map(lnk20, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))

    # save the SNR map
    snr_map = get_best_2comp_snr_mod(reg_final)
    hdr_save = hdr2D.copy()
    hdr_save.set(keyword='NOTES', value='Estimated peak signal-to-noise ratio', comment=None, before='DATE')
    hdr_save.set(keyword='BUNIT', value='unitless', comment=None, before=1)
    savename = make_save_name(paraRoot, paraDir, "SNR")
    save_map(snr_map, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))

    # create moment0 map
    modbest = get_best_2comp_model(reg_final)
    cube_mod = SpectralCube(data=modbest, wcs=copy(reg_final.ucube.pcubes['2'].wcs),
                            header=copy(reg_final.ucube.pcubes['2'].header))
    # make sure the spectral unit is in km/s before making moment maps
    cube_mod = cube_mod.with_spectral_unit('km/s', velocity_convention='radio')
    mom0_mod = cube_mod.moment0()
    savename = make_save_name(paraRoot, paraDir, "model_mom0")
    mom0_mod.write(savename, overwrite=True)
    logger.debug('{} saved.'.format(savename))

    # created masked mom0 map with model as the mask
    mom0 = UCube.get_masked_moment(cube=reg_final.ucube.cube, model=modbest, order=0, expand=20, mask=None)
    savename = make_save_name(paraRoot, paraDir, "mom0")
    mom0.write(savename, overwrite=True)
    logger.debug('{} saved.'.format(savename))

    # save reduced chi-squred maps
    # would be useful to check if 3rd component is needed
    '''
    savename = make_save_name(paraRoot, paraDir, "chi2red_final")
    chi_map = UCube.get_chisq(cube=reg_final.ucube.cube, model=modbest, expand=20, reduced=True, usemask=True,
                              mask=None)
    hdr_save = hdr2D.copy()
    hdr_save.set(keyword='NOTES', value='chi-squared values of the best 1- or 2-comp fit model',
                 comment=None, before='DATE')
    #hdr_save.set(keyword='BUNIT', value='', comment=None, before=1)
    save_map(chi_map, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))
    '''

    # save reduced chi-squred maps for 1 comp and 2 comp individually
    chiRed_1c = reg_final.ucube.get_reduced_chisq(1)
    chiRed_2c = reg_final.ucube.get_reduced_chisq(2)

    savename = make_save_name(paraRoot, paraDir, "chi2red_1c")
    hdr_save = hdr2D.copy()
    hdr_save.set(keyword='NOTES', value='reduced chi-squared values of the 1-comp model',
                 comment=None, before='DATE')
    #hdr_save.set(keyword='BUNIT', value='', comment=None, before=1)
    save_map(chiRed_1c, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))

    savename = make_save_name(paraRoot, paraDir, "chi2red_2c")
    hdr_save = hdr2D.copy()
    hdr_save.set(keyword='NOTES', value='reduced chi-squared values of the 2-comp model',
                 comment=None, before='DATE')
    #hdr_save.set(keyword='BUNIT', value='', comment=None, before=1)
    save_map(chiRed_2c, hdr_save, savename)
    logger.debug('{} saved.'.format(savename))

    return reg


def save_map(map, header, savename, overwrite=True):
    fits_map = fits.PrimaryHDU(data=map, header=header)
    fits_map.writeto(savename, overwrite=overwrite)

# =======================================================================================================================
# functions that facilitate


def get_2comp_wide_guesses(reg, window_hwidth=3.5, snr_min=3, savefit=True, planemask=None):
    if not hasattr(reg, 'ucube_res_cnv'):
        # fit the residual with the one component model if this has not already been done.
        try:
            fit_best_2comp_residual_cnv(reg, window_hwidth=window_hwidth, res_snr_cut=snr_min, savefit=savefit, planemask=planemask)
        except SNRMaskError:
            msg = "No valid pixel to refit in the residual cube for snr_min={}.".format(snr_min)
            raise SNRMaskError(msg)

    def get_mom_guesses(reg):
        # get moment guesses from no masking
        cube_res = get_best_2comp_residual_SpectralCube(reg, masked=False, window_hwidth=3.5)

        # find moment around where one component has been fitted
        pcube1 = reg.ucube.pcubes['1']
        vmap = medfilt2d(pcube1.parcube[0], kernel_size=3)  # median smooth within a 3x3 square
        moms_res = mmg.vmask_moments(cube_res, vmap=vmap, window_hwidth=3.5)
        gg = mmg.moment_guesses_1c(moms_res[0], moms_res[1], moms_res[2])
        # make sure the guesses falls within the limits

        mask = np.isfinite(moms_res[0])
        gg = gss_rf.refine_each_comp(gg, mask=mask)
        return gg

    if not hasattr(reg, 'ucube_res_cnv') or ('1' not in reg.ucube_res_cnv.pcubes) or not hasattr(
            reg.ucube_res_cnv.pcubes['1'], 'parcube'):
        # if there were no successful fit to the convolved cube, get moment map from no masking
        return get_mom_guesses(reg)

    else:
        # mask over where one component model is better than a no-signal (i.e., noise) model
        aic1v0_mask = reg.ucube_res_cnv.get_AICc_likelihood(1, 0) > 5

        if np.sum(aic1v0_mask) >= 1:
            logger.info("number of good fits to convolved residual: {}".format(np.sum(aic1v0_mask)))
            # if there are at least one well fitted pixel to the residual
            data_cnv = np.append(reg.ucube_res_cnv.pcubes['1'].parcube, reg.ucube_res_cnv.pcubes['1'].errcube, axis=0)
            preguess = copy(data_cnv)

            # set pixels that are better modelled as noise to nan
            preguess[:, ~aic1v0_mask] = np.nan

            # use the dialated mask as a footprint to interpolate the guesses
            gmask = binary_dilation(aic1v0_mask)
            guesses_final = gss_rf.guess_from_cnvpara(preguess, reg.ucube_res_cnv.cube.header, reg.ucube.cube.header,
                                                      mask=gmask)
        else:
            logger.info("no good fit from convolved guess, using the moment guess for the full-res refit instead")
            # get moment guesses without masking instead
            guesses_final = get_mom_guesses(reg)

        return guesses_final


def fit_best_2comp_residual_cnv(reg, window_hwidth=3.5, res_snr_cut=3, savefit=True, planemask=None):
    # fit the residual of the best fitted model (note, this approach may not hold well if the two-slab model
    # insufficiently at describing the observation. Luckily, however, this fit is only to provide initial guess for the
    # final fit)
    # the default window_hwidth = 3.5 is about half-way between the main hyperfine and the satellite
    # note res_snr_cut is only used for moment guess and not the actual recovery fit.

    # need a mechanism to make sure reg.ucube.pcubes['1'], reg.ucube.pcubes['2'] exists
    cube_res_cnv = get_best_2comp_residual_cnv(reg, masked=True, window_hwidth=window_hwidth, res_snr_cut=res_snr_cut)

    ncomp = 1

    # note: no further masking is applied to vmap, as we assume only pixels with good vlsr will be used
    if not hasattr(reg, 'ucube_cnv'):
        # if convolved cube was not used to produce initial guesses, use the full resolution 1-comp fit as the reference
        pcube1 = reg.ucube.pcubes['1']
        vmap = medfilt2d(pcube1.parcube[0], kernel_size=3)  # median smooth within a 3x3 square
        vmap = cnvtool.regrid(vmap, header1=get_skyheader(pcube1.header), header2=get_skyheader(cube_res_cnv.header))
    else:
        vmap = reg.ucube_cnv.pcubes['1'].parcube[0]

    import pyspeckit
    # use pcube moment estimate instead (it allows different windows for each pixel)
    pcube_res_cnv = pyspeckit.Cube(cube=cube_res_cnv)

    try:
        moms_res_cnv = mmg.window_moments(pcube_res_cnv, v_atpeak=vmap, window_hwidth=window_hwidth)
    except ValueError as e:
        raise SNRMaskError("There doesn't seem to be enough pixels to find the residual moments. {}".format(e))

    gg = mmg.moment_guesses_1c(moms_res_cnv[0], moms_res_cnv[1], moms_res_cnv[2])

    mask = np.isfinite(gg[0])
    gg = gss_rf.refine_each_comp(gg, mask=mask)

    mom0 = moms_res_cnv[0]
    if res_snr_cut > 0:
        rms = cube_res_cnv.mad_std(axis=0)
        snr = mom0 / rms
        maskmap = snr >= res_snr_cut
    else:
        maskmap = np.isfinite(mom0)

    if planemask is not None:
        if planemask.shape == maskmap.shape:
            maskmap = np.logical_and(maskmap, planemask)
        else:
            from reproject import reproject_interp
            # dilate the planemask to ensure no area lost when downsampling in repojecting
            planemask_regrid = binary_dilation(planemask, disk(2))
            planemask_regrid, _ = reproject_interp((planemask_regrid, reg.ucube.cube.wcs.celestial),
                                                   output_projection=cube_res_cnv.wcs.celestial,
                                                   shape_out=maskmap.shape)
            planemask_regrid = planemask_regrid > 0.5
            maskmap = np.logical_and(maskmap, planemask_regrid)

    reg.ucube_res_cnv = UCube.UltraCube(cube=cube_res_cnv, fittype=reg.fittype)

    if np.sum(maskmap) > 0:
        # note: snr is set to zero here becasue snr masking has already been pefromed once here
        try:
            reg.ucube_res_cnv.fit_cube(ncomp=[1], simpfit=False, snr_min=0.0, guesses=gg, maskmap=maskmap)
        except StartFitError:
            msg = "The first fitted pixel to the convovled residual cube did not yield a fit, likely due to a lack of signal or poor guesses."
            raise StartFitError(msg)
    else:
        msg = "No valid pixel to refit in the residual cube with snr_min={}. " \
              "Please consider trying a lower snr_min value".format(res_snr_cut)
        raise SNRMaskError(msg)

    # save the residual fit
    if savefit:
        oriParaPath = reg.ucube.paraPaths[str(ncomp)]
        savename = "{}_onBestResidual.fits".format(os.path.splitext(oriParaPath)[0])
        reg.ucube_res_cnv.save_fit(savename, ncomp)


def get_best_2comp_residual_cnv(reg, masked=True, window_hwidth=3.5, res_snr_cut=3):
    # convolved residual cube.If masked is True, only convolve over where 'excessive' residual is
    # above a peak SNR value of res_snr_cut masked

    cube_res_masked = get_best_2comp_residual_SpectralCube(reg, masked=masked, window_hwidth=window_hwidth,
                                                           res_snr_cut=res_snr_cut)

    if cube_res_masked.mask is None:
        msg = "Cube_res_masked does not have a mask for res_snr_cut={}. No residual refitting will be performed".format(res_snr_cut)
        logger.debug(msg)
        raise SNRMaskError(msg)
        return

    cube_res_cnv = cnvtool.convolve_sky_byfactor(cube_res_masked, factor=reg.cnv_factor, edgetrim_width=None,
                                                     snrmasked=False, iterrefine=False)

    return cube_res_cnv


def get_best_2comp_residual_SpectralCube(reg, masked=True, window_hwidth=3.5, res_snr_cut=5):
    # return residual cube as SpectralCube oobject
    # need a mechanism to make sure reg.ucube.pcubes['1'], reg.ucube.pcubes['2'] exists

    res_cube = get_best_2comp_residual(reg)
    best_res = res_cube._data
    cube_res = SpectralCube(data=best_res, wcs=copy(reg.ucube.pcubes['2'].wcs),
                            header=copy(reg.ucube.pcubes['2'].header))

    if masked and res_snr_cut > 0:
        best_rms = UCube.get_rms(res_cube._data)

        # calculate the peak SNR value of the best-fit residual over the main hyperfine components
        vmap = reg.ucube.pcubes['1'].parcube[0]
        # make want to double check that masked cube always masks out nan values
        res_main_hf = mmg.vmask_cube(res_cube, vmap, window_hwidth=window_hwidth)

        if res_main_hf.size > 1e7:
            # to avoid loading entire cube and stress the memory
            how = 'slice'
        else:
            # note: 'auto' currently returns slice for n > 1e8
            how = 'auto'

        # enable huge operations (note: not needed when "how" is chosen wisely, which it should be)
        res_main_hf.allow_huge_operations = True
        res_main_hf_snr = res_main_hf.max(axis=0, how=how).value / best_rms
        res_main_hf.allow_huge_operations = False

        # mask out residual with SNR values over the cut threshold
        mask_res = res_main_hf_snr > res_snr_cut
        if mask_res.sum() < 1:
            logger.debug("No pixel in the residual cube mask. No mask will be applied.")
            mask_res = binary_dilation(mask_res)
            cube_res_masked = cube_res.with_mask(~mask_res)
        else:
            # no masking
            cube_res_masked = cube_res
    else:
        # no masking
        cube_res_masked = cube_res

    cube_res_masked = cube_res_masked.with_spectral_unit(u.km / u.s, velocity_convention='radio')

    return cube_res_masked


def get_best_2comp_residual(reg):
    modbest = get_best_2comp_model(reg)
    best_res = reg.ucube.cube._data - modbest
    res_cube = SpectralCube(best_res, reg.ucube.cube.wcs)
    res_cube = res_cube.with_spectral_unit(u.km / u.s, velocity_convention='radio')
    return res_cube


def get_best_2comp_snr_mod(reg):
    modbest = get_best_2comp_model(reg)
    res_cube = get_best_2comp_residual(reg)
    best_rms = UCube.get_rms(res_cube._data)
    return np.nanmax(modbest, axis=0) / best_rms


def get_best_2comp_model(reg):
    # get the log-likelihood between the fits
    lnk10, lnk20, lnk21 = reg.ucube.get_all_lnk_maps(ncomp_max=2, rest_model_mask=True)

    mod1 = reg.ucube.pcubes['1'].get_modelcube()
    mod2 = reg.ucube.pcubes['2'].get_modelcube()

    # get the best model based on the calculated likelihood
    modbest = copy(mod1)
    modbest[:] = 0.0

    mask = lnk10 > 5
    modbest[:, mask] = mod1[:, mask]

    mask = np.logical_and(lnk21 > 5, lnk20 > 5)
    modbest[:, mask] = mod2[:, mask]

    return modbest


def replace_para(pcube, pcube_ref, mask, multicore=None):
    import multiprocessing

    # replace values in masked pixels with the reference values
    pcube.parcube[:, mask] = deepcopy(pcube_ref.parcube[:, mask])
    pcube.errcube[:, mask] = deepcopy(pcube_ref.errcube[:, mask])
    pcube.has_fit[mask] = deepcopy(pcube_ref.has_fit[mask])

    multicore = validate_n_cores(multicore)
    newmod = pcube_ref.get_modelcube(update=False, multicore=multicore)

    # Convert `pcube._modelcube` to a numpy array if it is currently a dask array
    if isinstance(pcube._modelcube, da.Array):
        pcube._modelcube = pcube._modelcube.compute()  # Load the full array into memory

    pcube._modelcube[:, mask] = deepcopy(newmod[:, mask])


def get_skyheader(cube_header):
    # a quick method to convert 3D header to 2D
    from astropy import wcs
    hdr = cube_header
    hdr2D = wcs.WCS(hdr).celestial.to_header()
    hdr2D['NAXIS'] = hdr2D['WCSAXES']
    hdr2D['NAXIS1'] = hdr['NAXIS1']
    hdr2D['NAXIS2'] = hdr['NAXIS2']
    return hdr2D