
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve

def expand_interpolate(data, ker_sig=1):
    # data should be a 3D array
    kernel = Gaussian2DKernel(ker_sig)  # large kernel size because regions outside the guesses are likely noisy

    data_smooth = data.copy()

    if data.ndim == 2:
        data_smooth = convolve(data, kernel, boundary='extend')
        qmask = np.isnan(data)
        data[qmask] = data_smooth[qmask]

    elif data.dim == 3:
        for i, gsm in enumerate(data_smooth):
            data_smooth[i] = convolve(data[i], kernel, boundary='extend')
            qmask = np.isnan(data[i])
            data[i][qmask] = data_smooth[i][qmask]

    return data


def iter_expand(data, mask):
    # expand the interpolation outwards, with increasingly larger kernals
    # until all pixels in the mask are filled

    def get_expand(data, mask):
        if data.ndim == 3:
            expand = np.sum(np.any(np.isnan(data), axis=0)[mask]) > 0
        elif data.ndim == 2:
            expand = np.sum(np.isnan(data)[mask]) > 0
        return expand

    expand = get_expand(data, mask)

    i = 0
    while expand:
        data = expand_interpolate(data, ker_sig= 1 +i)
        expand = get_expand(data, mask)
        i = i+ 1
    try:
        data[:, ~mask] = np.nan
    except IndexError:
        data[~mask] = np.nan

    return data