import numpy as np
from scipy.sparse import csr_matrix
from skimage.segmentation import find_boundaries


def get_matched_cells(arr1, arr2, mismatch_area):
    a = set((tuple(i) for i in arr1))
    b = set((tuple(i) for i in arr2))
    mismatch_pixel = list(b - a)
    if len(mismatch_pixel) <= mismatch_area:
        if len(mismatch_pixel) != 0:
            return np.array(list(a)), np.array(list(a & b))
        else:
            # diff = np.array(list(a - b))
            return np.array(list(a)), np.array(list(b))
    else:
        return False, False


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)),
                      shape=(data.max() + 1, data.size))


def get_indices_sparse(data):
    M = compute_M(data)
    return [np.unravel_index(row.data, data.shape) for row in M]


def list_remove(c_list, indexes):
    for index in sorted(indexes, reverse=True):
        del c_list[index]
    return c_list


def get_indexed_mask(mask, boundary):
    boundary = boundary * 1
    boundary_loc = np.where(boundary == 1)
    boundary[boundary_loc] = mask[boundary_loc]
    return boundary


def get_boundary(mask):
    mask_boundary = find_boundaries(mask)
    mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
    return mask_boundary_indexed


def get_mask(cell_list, shape):
    mask = np.zeros(shape)
    for cell_num in range(len(cell_list)):
        mask[tuple(cell_list[cell_num].T)] = cell_num + 1
    # print(get_cell_num(mask))
    return mask


def get_cell_num(mask):
    return len(np.unique(mask))


def get_matched_masks(mask_stack):
    """
    returns masks with matched cells and nuclei
    """
    matched_mask_stack = mask_stack.copy()
    whole_cell_mask = matched_mask_stack[0, :, :]  # todo: upgrade to multiple c channels
    nuclear_mask = matched_mask_stack[1, :, :]

    cell_coords = get_indices_sparse(whole_cell_mask)[1:]
    nucleus_coords = get_indices_sparse(nuclear_mask)[1:]

    cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
    nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))

    cell_matched_index_list = []
    nucleus_matched_index_list = []
    cell_matched_list = []
    nucleus_matched_list = []

    # print(len(cell_coords))
    # print(len(nucleus_coords))
    for i in range(len(cell_coords)):
        if len(cell_coords[i]) != 0:
            current_cell_coords = cell_coords[i]
            nuclear_search_num = np.unique(list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords)))
            # print(nuclear_search_num)
            for j in nuclear_search_num:
                if j != 0:
                    if (j - 1 not in nucleus_matched_index_list) and (i not in cell_matched_index_list):
                        whole_cell, nucleus = get_matched_cells(cell_coords[i], nucleus_coords[j - 1], mismatch_area=20)
                        if type(whole_cell) != bool:
                            cell_matched_list.append(whole_cell)
                            nucleus_matched_list.append(nucleus)
                            # cytoplasm_list.append(cytoplasm)
                            cell_matched_index_list.append(i)
                            nucleus_matched_index_list.append(j - 1)

    del cell_coords
    del nucleus_coords

    cell_matched_mask = get_mask(cell_matched_list, whole_cell_mask.shape)
    nuclear_matched_mask = get_mask(nucleus_matched_list, whole_cell_mask.shape)
    cell_membrane_mask = get_boundary(cell_matched_mask)
    nuclear_membrane_mask = get_boundary(nuclear_matched_mask)

    matched_mask_stack[0, :, :] = cell_matched_mask
    matched_mask_stack[1, :, :] = nuclear_matched_mask
    matched_mask_stack[2, :, :] = cell_membrane_mask
    matched_mask_stack[3, :, :] = nuclear_membrane_mask
    return matched_mask_stack


def get_mismatched_fraction(whole_cell_mask, nuclear_mask, cell_matched_mask, nuclear_matched_mask):
    whole_cell_mask_binary = np.sign(whole_cell_mask)
    nuclear_mask_binary = np.sign(nuclear_mask)
    cell_matched_mask_binary = np.sign(cell_matched_mask)
    nuclear_matched_mask_binary = np.sign(nuclear_matched_mask)
    total_area = np.sum(np.sign(whole_cell_mask_binary + nuclear_mask_binary))
    mismatched_area = np.sum(np.sign((nuclear_mask_binary - nuclear_matched_mask_binary) + (whole_cell_mask_binary - cell_matched_mask_binary)))
    mismatched_fraction = mismatched_area / total_area
    return mismatched_fraction


def save_mismatch_fraction(out_path, whole_cell_mask, nuclear_mask, cell_matched_mask, nuclear_matched_mask):
    mismatched_fraction = get_mismatched_fraction(whole_cell_mask, nuclear_mask, cell_matched_mask, nuclear_matched_mask)
    np.savetxt(out_path, [mismatched_fraction])

