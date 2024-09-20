__author__ = 'mcychen'

from skimage.segmentation import watershed
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import binary_dilation, disk, medial_axis

#======================================================================================================================#
from .mufasa_log import get_logger
logger = get_logger(__name__)
#======================================================================================================================#
# utility functions to help divide maps in different ways


def dist_divide(seeds, weights=None, return_nmarkers=False):

    #label the peaks as distincts markers
    markers, n_markers = ndi.label(seeds)
    logger.debug("number of markers: {}".format(n_markers))

    # map the distance between the peaks
    dist = ndi.distance_transform_edt(~seeds)

    if weights is None:
        labels = watershed(dist, markers)
    else:
        labels = watershed(dist*weights, markers)

    if return_nmarkers:
        return labels, n_markers
    else:
        return labels


def watershed_divide(image, seeds):

    markers, n_markers = ndi.label(seeds)
    logger.debug("number of markers: {}".format(n_markers))
    return watershed(image, markers)
