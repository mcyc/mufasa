
import numpy as np


def find_maxref_neighbor(mask, data, ref):
    # find pixel of a neighbour with the highest reference value
    nearest_refs = []
    highest_refs = []
    maxref_data = []
    highest_coords = []

    # Step 1: Get coordinates of mask
    true_pixel_coords = np.argwhere(mask)

    # Step 2: For each pixel within the mask, find its valid neighbors
    for x, y in true_pixel_coords:
        neighbors = get_valid_neighbors(mask, ref, x, y)

        if neighbors:
            # Step 3: Find the nearest (any neighbor will do since they're immediate)
            nearest_ref = neighbors[0][1]  # Get the value of the first neighbor

            # Step 4: Find the pixel with the highest reference value among the neighbors
            highest_neighbor = max(neighbors, key=lambda x: x[1])  # Find the highest neighbor

            highest_ref = highest_neighbor[1]
            highest_coord = highest_neighbor[0]  # Get the coordinates of the highest ref pixel

            # Step 5: Get the data value at position of the highest ref neighbor
            maxref_value = data[highest_coord]
        else:
            # No valid neighbors
            nearest_ref = None
            highest_ref = None
            maxref_value = None
            highest_coord = None

        nearest_refs.append(nearest_ref)
        highest_refs.append(highest_ref)
        highest_coords.append(highest_coord)
        maxref_data.append(maxref_value)

    return nearest_refs, highest_refs, maxref_data, highest_coords


def get_valid_neighbors(mask, data, x, y):
    neighbors = []
    rows, cols = mask.shape

    # Check up, down, left, right neighbors
    if x > 0 and not mask[x - 1, y]:  # up
        neighbors.append(((x - 1, y), data[x - 1, y]))
    if x < rows - 1 and not mask[x + 1, y]:  # down
        neighbors.append(((x + 1, y), data[x + 1, y]))
    if y > 0 and not mask[x, y - 1]:  # left
        neighbors.append(((x, y - 1), data[x, y - 1]))
    if y < cols - 1 and not mask[x, y + 1]:  # right
        neighbors.append(((x, y + 1), data[x, y + 1]))

    return neighbors
