"""
Utilities for working with .fits files, including functions adapted from
FITS_tools to reduce dependencies on the aging package.
"""


def downsample_header(header, factor, axis):
    """
    Downsample a FITS header along a specified axis.

    This function downsamples the provided FITS header along the specified axis
    using the FITS convention for axis numbering. The `CDELT` and `CRPIX`
    keywords are adjusted accordingly to reflect the downsampling.

    Parameters
    ----------
    header : astropy.io.fits.Header
        The FITS header to be downsampled. This is typically obtained from a
        FITS file.
    factor : float
        The downsampling factor. Must be greater than 0.
    axis : int
        The axis along which to apply the downsampling. Follows the FITS
        convention for axis numbering (1-based indexing).

    Returns
    -------
    astropy.io.fits.Header
        A copy of the FITS header with adjusted `CDELT` and `CRPIX` values to
        account for the downsampling.

    Notes
    -----
    - The function modifies a copy of the input header, leaving the original
      header unchanged.
    - The code here is borrowed from `FITS_tools.downsample`.

    Examples
    --------
    >>> from astropy.io import fits
    >>> header = fits.Header()
    >>> header['CDELT1'] = 0.1
    >>> header['CRPIX1'] = 50
    >>> downsampled_header = downsample_header(header, factor=2, axis=1)
    >>> print(downsampled_header['CDELT1'], downsampled_header['CRPIX1'])
    0.2 25.5
    """
    header = header.copy()

    cd = 'CDELT{0:d}'.format(axis)
    cp = 'CRPIX{0:d}'.format(axis)
    scalefactor = 1./factor
    header[cp] = (header[cp]-1)*scalefactor + scalefactor/2. + 0.5
    header[cd] = header[cd]*factor

    return header


def get_pixel_mapping(header1, header2):
    """
    Compute the pixel mapping between two FITS headers or WCS objects.

    This function determines how pixel coordinates in one FITS header
    or WCS (`header1`) map to pixel coordinates in another (`header2`).
    It returns a grid that describes the transformation in terms of
    world coordinates.

    Parameters
    ----------
    header1 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS corresponding to the input image whose pixel
        coordinates are being transformed.
    header2 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
        The header or WCS corresponding to the output image onto which
        the input image will be interpolated.

    Returns
    -------
    grid : `~numpy.ndarray`
        A 2D array of shape `(2, n, m)` where `n` and `m` are the dimensions
        of `header2`. The array contains the mapped pixel coordinates from
        `header1` in the following order:
        - `grid[0, :, :]`: The y-coordinates in `header1`.
        - `grid[1, :, :]`: The x-coordinates in `header1`.

    Raises
    ------
    TypeError
        If either `header1` or `header2` is not an instance of
        `astropy.io.fits.Header` or `astropy.wcs.WCS`.
    NotImplementedError
        If the coordinate system type (`CTYPE`) in the headers is not
        recognized or if unit conversions between coordinate systems
        are not implemented.

    Notes
    -----
    - The function supports transformations between images that share
      compatible celestial coordinate systems, such as `GLON`/`GLAT`
      and `RA`/`DEC`.
    - If the coordinate systems differ, they will be transformed to
      align before computing the pixel mapping.
    - The code here is borrowed from `FITS_tools.downsample`.

    Examples
    --------
    >>> from astropy.io import fits
    >>> from astropy.wcs import WCS
    >>> header1 = fits.Header()  # Define input header or WCS
    >>> header2 = fits.Header()  # Define output header or WCS
    >>> grid = get_pixel_mapping(header1, header2)
    >>> print(grid.shape)
    (2, n, m)  # Example output shape corresponding to header2 dimensions
    """
    wcs1 = _load_wcs_from_header(header1)
    wcs2 = _load_wcs_from_header(header2)

    if not all([w1 == w2 for w1, w2 in zip(wcs1.wcs.ctype, wcs2.wcs.ctype)]):
        allowed_coords = ('GLON', 'GLAT', 'RA', 'DEC')
        if all([(any(word in w1 for word in allowed_coords) and
                 any(word in w2 for word in allowed_coords))
                for w1, w2 in zip(wcs1.wcs.ctype, wcs2.wcs.ctype)]):
            csys1 = _ctype_to_csys(wcs1.wcs)
            csys2 = _ctype_to_csys(wcs2.wcs)
            convert_coordinates = True
        else:
            raise NotImplementedError(
                "Unit conversions between {0} and {1} have not yet been implemented."
                .format(wcs1.wcs.ctype, wcs2.wcs.ctype)
            )
    else:
        convert_coordinates = False

    outshape = [wcs2.naxis2, wcs2.naxis1]
    yy2, xx2 = np.indices(outshape)

    lon2, lat2 = wcs2.wcs_pix2world(xx2, yy2, 0)

    if convert_coordinates:
        C2 = coordinates.SkyCoord(lon2, lat2, unit=(u.deg, u.deg), frame=csys2)
        C1 = C2.transform_to(csys1)
        lon2, lat2 = C1.spherical.lon.deg, C1.spherical.lat.deg

    xx1, yy1 = wcs1.wcs_world2pix(lon2, lat2, 0)
    grid = np.array([yy1.reshape(outshape), xx1.reshape(outshape)])

    return grid
