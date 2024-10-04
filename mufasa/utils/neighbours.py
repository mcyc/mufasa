
import numpy as np


def maxref_neighbor_coords(mask, ref, fill_coord=(0, 0)):
    # find pixel of a neighbour with the highest reference value
    highest_coords = []

    # Get coordinates of mask
    true_pixel_coords = np.argwhere(mask)

    # For each pixel within the mask, find its valid neighbors
    for x, y in true_pixel_coords:
        neighbors = get_valid_neighbors(mask, ref, x, y)

        if neighbors:
            # Find the pixel with the highest reference value among the neighbors
            highest_neighbor = max(neighbors, key=lambda x: x[1])  # Find the highest neighbor
            highest_coord = highest_neighbor[0]  # Get the coordinates of the highest ref pixel
        else:
            highest_coord = fill_coord

        highest_coords.append(highest_coord)

    return highest_coords


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
