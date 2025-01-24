
import numpy as np
import warnings
from skimage.morphology import disk

try:
    from skimage.morphology import footprint_rectangle
except ImportError:
    warnings.warn(
        "footprint_rectangle is not available in this version of scikit-image. "
        "Using a fallback implementation.",
        ImportWarning
    )
    def footprint_rectangle(height, width):
        return np.ones((height, width), dtype=bool)

def maxref_neighbor_coords(mask, ref, fill_coord=(0, 0), structure=None, centre=None):
    # find pixel of a neighbour with the highest reference value
    highest_coords = []

    # Get coordinates of mask
    true_pixel_coords = np.argwhere(mask)

    if structure is None:
        structure, centre = square_neighbour(r=1, return_centre=True)
    else:
        if isinstance(structure, str):
            if structure == '4-connect':
                # 4 nearest neighbours
                structure, centre = disk_neighbour(1, return_centre=True)
            elif structure == '8-connect':
                # 8 nearest neighbours
                structure, centre = square_neighbour(r=1, return_centre=True)

    if centre is None:
        centre = structure.shape
        centre = ( np.ceil(centre[0]/2).astype(int) - 1 , np.ceil(centre[1]/2).astype(int) - 1 )

    # For each pixel within the mask, find its valid neighbors
    for y, x in true_pixel_coords:
        neighbors = get_valid_neighbors(ref, x, y, structure=structure, centre=centre)

        if neighbors:
            # Find the pixel with the highest reference value among the neighbors
            highest_neighbor = max(neighbors, key=lambda x: x[1])  # Find the highest neighbor
            highest_coord = highest_neighbor[0]  # Get the coordinates of the highest ref pixel
        else:
            highest_coord = fill_coord

        highest_coords.append(highest_coord)

    return highest_coords

def get_valid_neighbors(data, x, y, structure, centre):
    neighbors = []

    ycoord, xcoord = get_neighbor_coord(structure, x,y, centre[1], centre[0])

    for y, x in zip(ycoord, xcoord):
        try:
            neighbors.append(((y, x), data[y, x]))
        except IndexError:
            continue

    return neighbors

def get_neighbor_coord(struct, x, y, x_ref, y_ref):
    idx = np.where(struct)
    ycoord = idx[0] - y_ref + y
    xcoord = idx[1] - x_ref + x
    return (ycoord, xcoord)

def disk_neighbour(r, return_centre=False):
    struct = disk(r, dtype='bool')
    struct[r, r] = False
    if return_centre:
        return struct, (r,r)
    else:
        return struct

def square_neighbour(r, return_centre=False):
    # r is the degree of seperation from
    width = r*2+1
    struct = footprint_rectangle((width, width), dtype='bool')
    struct[r, r] = False
    if return_centre:
        return struct, (r,r)
    else:
        return struct