"""
The `mufasa.slab_sort` module provides utilities for sorting and refining two-component spectral data, including
functions for swapping components based on linewidth or distance metrics relative
to reference maps. These tools are intended to facilitate multi-component spectral
analysis by improving data consistency and organization.
"""

from __future__ import print_function
__author__ = 'mcychen'

#=======================================================================================================================

import numpy as np
from scipy.ndimage.filters import median_filter

#=======================================================================================================================

def sort_2comp(data, method="linewidth_wide"):

    if method == "linewidth_narrow":
        # move the narrower linewidth component to the front
        swapmask = data[5] < data[1]

    elif method == "linewidth_wide":
        # move the wider linewidth component to the front
        swapmask = data[5] > data[1]

    return mask_swap_2comp(data, swapmask)


def quick_2comp_sort(data, filtsize=2):
    # use median filtered vlsr & sigma maps as a velocity reference to sort the two components

    # arange the maps so the component with the least vlsr errors is the first component
    swapmask = data[8] > data[12]
    data = mask_swap_2comp(data, swapmask)

    # the use the vlsr error in the first component as the reference and sort the component based on their similarities
    # to this reference (similary bright structures should have similar errors)
    ref = median_filter(data[8], size=(filtsize, filtsize))
    swapmask = np.abs(data[8] - ref) > np.abs(data[12] - ref)
    data = mask_swap_2comp(data, swapmask)

    def dist_metric(p1, p2):
        # use the first map (the one that should have the smallest error, hense more reliable) to compute
        #  distance metric based on their similarities to the median filtered quantity
        p_refa = median_filter(p1, size=(filtsize, filtsize))
        p_refb = median_filter(p2, size=(filtsize, filtsize))

        # distance of the current arangment to the median
        del_pa = np.abs(p1 - p_refa) #+ np.abs(p2 - p_refb)
        #del_pa = np.hypot(np.abs(p1 - p_refa), np.abs(p2 - p_refb))

        # distance of the swapped arangment to the median
        del_pb = np.abs(p2 - p_refa) #+ np.abs(p1 - p_refb)
        #del_pb = np.hypot(np.abs(p2 - p_refa),np.abs(p1 - p_refb))
        return del_pa, del_pb

    dist_va, dist_vb = dist_metric(data[0], data[4])
    dist_siga, dist_sigb = dist_metric(data[1], data[5])

    #swapmask = dist_va > dist_vb
    # use both the vlsr and the sigma as a distance metric
    swapmask = np.hypot(dist_va, dist_siga) > np.hypot(dist_vb, dist_sigb)

    data= mask_swap_2comp(data, swapmask)

    return data


#=======================================================================================================================

def mask_swap_2comp(data, swapmask):
    # swap data over the mask
    data= data.copy()
    data[0:4,swapmask], data[4:8,swapmask] = data[4:8,swapmask], data[0:4,swapmask]
    data[8:12,swapmask], data[12:16,swapmask] = data[12:16,swapmask], data[8:12,swapmask]
    return data


#=======================================================================================================================

def refmap_2c_mask(pmaps, refmaps, method="v_n_sig"):
    # return masks where 1st component is further from the ref maps than the 2nd by the distance metric

    del_vlsr = distance_metric(pmaps[0], pmaps[4], refmaps[0])
    del_sigv = distance_metric(pmaps[1], pmaps[5], refmaps[1])

    if method == 'vlsr':
        return del_vlsr[0] > del_vlsr[1]

    elif method == 'sigv':
        return del_sigv[0] > del_sigv[1]

    elif method == "v_n_sig":
        return np.hypot(del_vlsr[0], del_sigv[0]) > np.hypot(del_vlsr[1], del_sigv[1])

    else:
        raise Exception("The method provided is not valid!")
        return None


def distance_metric(p1, p2, p_refa):
    # distance of 1st component to the reference
    del_pa = np.abs(p1 - p_refa)
    # distance of 2st component to the reference
    del_pb = np.abs(p2 - p_refa)

    return del_pa, del_pb


#=======================================================================================================================