import numpy as np
from astropy.io import fits
import pandas as pd
import numpy as np

def read(filename, header=True, vrange=None, verr_min=5):
    para, hdr = fits.getdata(filename, header=header)
    dataframe = make_dataframe(para, vrange, verr_min=verr_min)

    if header:
        return dataframe, hdr
    else:
        return dataframe


def make_dataframe(para_2c, vrange=None, verr_min=5):
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
        'vlsr': [],
        'x_crd': [],
        'y_crd': [],
        'eVlsr': [],
        'sigv': [],
        'tex': [],
        'tau': [],
        'comp_i': []
    }

    for i in range(ncomp):
        # loop through the two components
        mask = np.isfinite(para_2c[i])

        # impose an error threshold
        mask = np.logical_and(mask, para_2c[i * 4 + 8] < verr_min)

        # impose velocity range threshold
        mask = np.logical_and(mask, para_2c[i * 4] < v_max)
        mask = np.logical_and(mask, para_2c[i * 4] > v_min)

        data['vlsr'] += para_2c[i * 4][mask].tolist()
        data['x_crd'] += x_crds[mask].tolist()
        data['y_crd'] += y_crds[mask].tolist()

        data['eVlsr'] += para_2c[i * 4 + 8][mask].tolist()
        data['sigv'] += para_2c[i * 4 + 1][mask].tolist()
        data['tex'] += para_2c[i * 4 + 2][mask].tolist()
        data['tau'] += para_2c[i * 4 + 3][mask].tolist()
        data['comp_i'] += [i] * np.sum(mask) # comp_i is there to break the xy-degerency when adding more info

    return pd.DataFrame(data)

def assign_to_dataframe(dataframe, new_map, comp_i):
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


