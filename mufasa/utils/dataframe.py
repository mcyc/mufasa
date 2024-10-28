import numpy as np
from astropy.io import fits
import pandas as pd

def read(filename, header=True, vrange=None, verr_thres=5):
    """
    Read a FITS file and convert the data to a pandas DataFrame, optionally including the header.

    Parameters
    ----------
    filename : str
        Path to the .fits file containing the MUFASA output parameter data.
    header : bool, optional
        Whether to include the header in the output. Default is True.
    vrange : tuple of float, optional
        Velocity range to filter the data, specified as (v_min, v_max) in km/s. Default is None, which includes all data.
    verr_thres : float, optional
        Velocity error threshold (in km/s) to exclude data points with higher errors. Default is 5.

    Returns
    -------
    dataframe : pandas.DataFrame
        A DataFrame containing the parameter data, including columns such as 'vlsr', 'x_crd', 'y_crd', 'eVlsr', 'sigv',
         'tex', 'tau', and 'comp_i'.
    hdr : astropy.io.fits.Header, optional
        The header from the FITS file, included if `header` is True.

    Notes
    -----
    The function reads a FITS file, converts the parameter data into a structured DataFrame, and applies the specified
     velocity and error thresholds if provided.
    """
    para, hdr = fits.getdata(filename, header=header)
    dataframe = make_dataframe(para, vrange, verr_thres=verr_thres)

    if header:
        return dataframe, hdr
    else:
        return dataframe


def make_dataframe(para_2c, vrange=None, verr_thres=5):
    """
    Create a DataFrame from a 3D parameter array, applying optional velocity and error thresholds.

    Parameters
    ----------
    para_2c : numpy.ndarray
        A 3D array containing parameter values across two spatial dimensions and one parameter axis.
    vrange : tuple of float, optional
        Velocity range to filter data, specified as (v_min, v_max) in km/s. Default is None, which includes all data.
    verr_thres : float, optional
        Velocity error threshold (in km/s) to exclude data points with higher errors. Default is 5.

    Returns
    -------
    dataframe : pandas.DataFrame
        A DataFrame containing extracted and filtered data, with columns such as 'vlsr', 'x_crd', 'y_crd', 'eVlsr',
        'sigv', 'tex', 'tau', and 'comp_i'.

    Notes
    -----
    The function extracts relevant parameters from `para_2c`, applies the specified velocity and error thresholds,
    and structures the data into a pandas DataFrame.
    """

    ncomp = int(para_2c.shape[0] / 8)

    if vrange is None:
        v_min, v_max = None, None
    else:
        v_min, v_max = vrange

    if v_min is None:
        v_min = -1.0 * np.inf
    if v_max is None:
        v_max = np.inf

    # get the coordinate grid
    nz, ny, nx = para_2c.shape
    x_crds, y_crds = np.meshgrid(range(nx), range(ny), indexing='xy')

    data = {
        'x_crd': [], 'y_crd': [],
        'vlsr': [], 'sigv': [], 'tex': [], 'tau': [],
        'eVlsr': [], 'eSigv': [], 'eTex': [], 'eTau':[],
        'comp_i': []
    }

    for i in range(ncomp):
        # loop through the two components
        mask = np.isfinite(para_2c[i])

        # impose an error threshold
        mask = np.logical_and(mask, para_2c[i * 4 + 8] < verr_thres)

        # impose velocity range threshold
        mask = np.logical_and(mask, para_2c[i * 4] < v_max)
        mask = np.logical_and(mask, para_2c[i * 4] > v_min)

        data['x_crd'] += x_crds[mask].tolist()
        data['y_crd'] += y_crds[mask].tolist()

        data['vlsr'] += para_2c[i * 4][mask].tolist()
        data['sigv'] += para_2c[i * 4 + 1][mask].tolist()
        data['tex'] += para_2c[i * 4 + 2][mask].tolist()
        data['tau'] += para_2c[i * 4 + 3][mask].tolist()

        data['eVlsr'] += para_2c[i * 4 + 8][mask].tolist()
        data['eSigv'] += para_2c[i * 4 + 9][mask].tolist()
        data['eTex'] += para_2c[i * 4 + 10][mask].tolist()
        data['eTau'] += para_2c[i * 4 + 11][mask].tolist()

        data['comp_i'] += [i] * np.sum(mask) # comp_i is there to break the xy-degeneracy when adding more info

    return pd.DataFrame(data)


def assign_to_dataframe(dataframe, new_map, comp_i):
    """
    Assign values from a new data array to an existing DataFrame based on spatial coordinates and component index.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The main DataFrame containing spatial and component information.
    new_map : numpy.ndarray
        Array of new values to be assigned to the DataFrame, with shape (j, k).
    comp_i : int
        The component index associated with the new values.

    Returns
    -------
    dataframe : pandas.DataFrame
        Updated DataFrame with the new values assigned in a 'new_value' column.

    Notes
    -----
    This function merges the new values with the input DataFrame based on the x and y spatial coordinates and `comp_i`.
    """
    # Generate x and y coordinate grid for the new_map array (shape (j, k))
    j, k = new_map.shape
    x_crds, y_crds = np.meshgrid(range(k), range(j), indexing='xy')

    # Flatten the new_map and coordinate grids to create a DataFrame with the constant comp_i
    new_data = pd.DataFrame({
        'x_crd': x_crds.flatten(),
        'y_crd': y_crds.flatten(),
        'comp_i': comp_i,  # Add the constant comp_i to match with the main dataframe
        'new_value': new_map.flatten()
    })

    # Merge the new values based on 'x_crd', 'y_crd', and 'comp_i' with the input dataframe
    dataframe = dataframe.merge(new_data, on=['x_crd', 'y_crd', 'comp_i'], how='left')

    return dataframe
