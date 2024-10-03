from typing import Tuple
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from skimage.segmentation import find_boundaries
# from numba import njit


Image = np.ndarray

"""
Package functions that repair and generate matched cell, nuclear,
cell membrane and nuclear membrane segmentation masks
Author: Haoran Chen and Ted Zhang
Version: 2.0
09/2024
"""


def get_matched_cells(cell_arr, cell_membrane_arr, nuclear_arr, mismatch_repair):
    a = set((tuple(i) for i in cell_arr))
    b = set((tuple(i) for i in cell_membrane_arr))
    c = set((tuple(i) for i in nuclear_arr))
    d = a - b
    # remove cell membrane from cell
    mismatch_pixel_num = len(list(c - d))
    mismatch_fraction = len(list(c - d)) / len(list(c))
    if not mismatch_repair:
        if mismatch_pixel_num == 0:
            return np.array(list(a)), np.array(list(c)), 0
        else:
            return False, False, False
    else:
        if mismatch_pixel_num < len(c):
            return np.array(list(a)), np.array(list(d & c)), mismatch_fraction
        else:
            return False, False, False


def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


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


def get_boundary(mask: Image):
    mask_boundary = find_boundaries(mask)
    mask_boundary_indexed = get_indexed_mask(mask, mask_boundary)
    return mask_boundary_indexed


def get_mask(cell_list, shape: Tuple[int]):
    mask = np.zeros(shape)
    for cell_num in range(len(cell_list)):
        mask[tuple(cell_list[cell_num].T)] = cell_num + 1
    return mask


def get_cell_num(mask: Image):
    return len(np.unique(mask))


def get_mismatched_fraction(
    whole_cell_mask: Image,
    nuclear_mask: Image,
    cell_matched_mask: Image,
    nuclear_matched_mask: Image,
) -> float:
    whole_cell_mask_binary = np.sign(whole_cell_mask)
    nuclear_mask_binary = np.sign(nuclear_mask)
    cell_matched_mask_binary = np.sign(cell_matched_mask)
    nuclear_matched_mask_binary = np.sign(nuclear_matched_mask)
    total_area = np.sum(np.sign(whole_cell_mask_binary + nuclear_mask_binary))
    mismatched_area = np.sum(
        np.sign(
            (nuclear_mask_binary - nuclear_matched_mask_binary)
            + (whole_cell_mask_binary - cell_matched_mask_binary)
        )
    )
    mismatched_fraction = mismatched_area / total_area
    return mismatched_fraction


def get_fraction_matched_cells(
    whole_cell_mask: Image, nuclear_mask: Image, cell_matched_mask: Image
) -> float:
    matched_cell_num = len(np.unique(cell_matched_mask)) - 1
    total_cell_num = len(np.unique(whole_cell_mask)) - 1
    total_nuclei_num = len(np.unique(nuclear_mask)) - 1
    mismatched_cell_num = total_cell_num - matched_cell_num
    mismatched_nuclei_num = total_nuclei_num - matched_cell_num
    fraction_matched_cells = matched_cell_num / (
        mismatched_cell_num + mismatched_nuclei_num + matched_cell_num
    )
    return fraction_matched_cells

def get_boundary_2(mask: Image):
    mask_boundary = find_boundaries(mask)
    mask_boundary_indexed = mask_boundary * mask
    return mask_boundary_indexed

def get_matched_masks_optimized(mask_stack: Image, do_mismatch_repair: bool) -> Tuple[Image, float]:
    '''
    matches cells and nuclei and returns the matched masks --- optimized version

    Args:
    mask_stack: Image
        the mask stack with cell, nuclear, cell membrane and nuclear membrane masks
    do_mismatch_repair: bool
        whether to repair mismatched cells

    Returns:
    matched_mask_stack: Image
    '''
       
    matched_mask_stack = mask_stack.copy()
    whole_cell_mask = matched_mask_stack[0, :, :]
    nuclear_mask = matched_mask_stack[1, :, :]
    cell_membrane_mask = get_boundary(whole_cell_mask)

    # Get unique labels, excluding background (label 0)
    cell_labels = np.unique(whole_cell_mask)
    nuclear_labels = np.unique(nuclear_mask)
    cell_labels = cell_labels[cell_labels != 0]
    nuclear_labels = nuclear_labels[nuclear_labels != 0]
    num_cells = len(cell_labels)
    num_nuclei = len(nuclear_labels)

    # Create mapping from labels to indices
    cell_label_to_index = {label: idx for idx, label in enumerate(cell_labels)}
    nuclear_label_to_index = {label: idx for idx, label in enumerate(nuclear_labels)}
    index_to_cell_label = {idx: label for idx, label in enumerate(cell_labels)}
    index_to_nuclear_label = {idx: label for idx, label in enumerate(nuclear_labels)}

    # Flatten masks and identify overlapping pixels
    cell_mask_flat = whole_cell_mask.ravel()
    nuclear_mask_flat = nuclear_mask.ravel()
    valid_pixels = (cell_mask_flat > 0) & (nuclear_mask_flat > 0)

    cell_labels_at_pixels = cell_mask_flat[valid_pixels]
    nuclear_labels_at_pixels = nuclear_mask_flat[valid_pixels]
    cell_indices = np.array([cell_label_to_index[label] for label in cell_labels_at_pixels])
    nuclear_indices = np.array([nuclear_label_to_index[label] for label in nuclear_labels_at_pixels])

    # Build overlap matrix using sparse representation
    data = np.ones_like(cell_indices, dtype=np.int32)
    overlap_matrix = coo_matrix(
        (data, (cell_indices, nuclear_indices)),
        shape=(num_cells, num_nuclei)
    ).tocsr()

    # Compute total pixels for each cell and nucleus
    cell_sizes = np.bincount(cell_mask_flat, minlength=whole_cell_mask.max() + 1)
    cell_sizes = cell_sizes[cell_labels]
    nuclear_sizes = np.bincount(nuclear_mask_flat, minlength=nuclear_mask.max() + 1)
    nuclear_sizes = nuclear_sizes[nuclear_labels]

    # Initialize arrays to store the best match for each cell
    best_nucleus_indices = np.full(num_cells, -1, dtype=int)
    best_overlap_fractions = np.zeros(num_cells, dtype=float)

    # For each cell, find the nucleus with the maximum overlap fraction
    for cell_idx in range(num_cells):
        # Get the slice of the overlap matrix corresponding to the current cell
        start_ptr = overlap_matrix.indptr[cell_idx]
        end_ptr = overlap_matrix.indptr[cell_idx + 1]
        nucleus_indices = overlap_matrix.indices[start_ptr:end_ptr]
        overlap_counts = overlap_matrix.data[start_ptr:end_ptr]

        if len(nucleus_indices) == 0:
            continue  # No overlapping nuclei for this cell

        # Compute overlap fractions for this cell
        cell_size = cell_sizes[cell_idx]
        overlap_fractions = overlap_counts / cell_size

        # Find the nucleus with the maximum overlap fraction
        max_idx = np.argmax(overlap_fractions)
        best_nucleus_idx = nucleus_indices[max_idx]
        best_overlap_fraction = overlap_fractions[max_idx]

        best_nucleus_indices[cell_idx] = best_nucleus_idx
        best_overlap_fractions[cell_idx] = best_overlap_fraction

    # Determine matches based on overlap fractions and mismatch repair flag
    matched_cell_indices = []
    matched_nucleus_indices = []
    used_nuclei = set()

    for cell_idx in range(num_cells):
        nucleus_idx = best_nucleus_indices[cell_idx]
        if nucleus_idx == -1:
            continue  # No matching nucleus

        mismatch_fraction = 1 - best_overlap_fractions[cell_idx]
        if not do_mismatch_repair and mismatch_fraction > 0:
            continue  # Skip mismatched cells if not repairing

        if nucleus_idx in used_nuclei:
            continue  # Nucleus already matched to another cell

        matched_cell_indices.append(cell_idx)
        matched_nucleus_indices.append(nucleus_idx)
        used_nuclei.add(nucleus_idx)

    # Create matched masks
    cell_matched_mask = np.zeros_like(whole_cell_mask)
    nuclear_matched_mask = np.zeros_like(nuclear_mask)

    for cell_idx, nucleus_idx in zip(matched_cell_indices, matched_nucleus_indices):
        cell_label = index_to_cell_label[cell_idx]
        nucleus_label = index_to_nuclear_label[nucleus_idx]
        cell_pixels = (whole_cell_mask == cell_label)
        nucleus_pixels = (nuclear_mask == nucleus_label)

        if do_mismatch_repair:
            # Keep only overlapping pixels
            matched_nucleus_pixels = nucleus_pixels & cell_pixels
        else:
            matched_nucleus_pixels = nucleus_pixels

        cell_matched_mask[cell_pixels] = cell_label
        nuclear_matched_mask[matched_nucleus_pixels] = nucleus_label

    # Generate boundary masks
    cell_membrane_mask = get_boundary(cell_matched_mask)
    nuclear_membrane_mask = get_boundary(nuclear_matched_mask)

    # Compute fraction of matched cells
    if do_mismatch_repair:
        fraction_matched_cells = 1.0
    else:
        matched_cell_num = len(matched_cell_indices)
        total_cell_num = num_cells
        total_nuclei_num = num_nuclei
        mismatched_cell_num = total_cell_num - matched_cell_num
        mismatched_nuclei_num = total_nuclei_num - len(used_nuclei)
        fraction_matched_cells = matched_cell_num / (
            mismatched_cell_num + mismatched_nuclei_num + matched_cell_num
        )

    matched_mask_stack[0, :, :] = cell_matched_mask
    matched_mask_stack[1, :, :] = nuclear_matched_mask
    matched_mask_stack[2, :, :] = cell_membrane_mask
    matched_mask_stack[3, :, :] = nuclear_membrane_mask
    return matched_mask_stack, fraction_matched_cells

def get_matched_masks(mask_stack: Image, do_mismatch_repair: bool) -> Tuple[Image, float]:
    """
    returns masks with matched cells and nuclei
    """
    matched_mask_stack = mask_stack.copy()
    whole_cell_mask = matched_mask_stack[0, :, :]
    nuclear_mask = matched_mask_stack[1, :, :]
    cell_membrane_mask = get_boundary(whole_cell_mask)

    cell_coords = get_indices_sparse(whole_cell_mask)[1:]
    nucleus_coords = get_indices_sparse(nuclear_mask)[1:]
    cell_membrane_coords = get_indices_sparse(cell_membrane_mask)[1:]

    # # Get unique labels, excluding background (label 0)
    # cell_labels = np.unique(whole_cell_mask)
    # nuclear_labels = np.unique(nuclear_mask)
    # cell_labels = cell_labels[cell_labels != 0]
    # nuclear_labels = nuclear_labels[nuclear_labels != 0]

    # Create mapping from labels to indices
    # cell_label_to_index = {label: idx for idx, label in enumerate(cell_labels)}
    # nuclear_label_to_index = {label: idx for idx, label in enumerate(nuclear_labels)}

    # # Flatten masks and identify overlapping pixels
    # cell_mask_flat = whole_cell_mask.ravel()
    # nuclear_mask_flat = nuclear_mask.ravel()
    # valid_pixels = (cell_mask_flat > 0) & (nuclear_mask_flat > 0)

    # cell_labels_at_pixels = cell_mask_flat[valid_pixels]
    # nuclear_labels_at_pixels = nuclear_mask_flat[valid_pixels]
    # cell_indices = np.array([cell_label_to_index[label] for label in cell_labels_at_pixels])
    # nuclear_indices = np.array([nuclear_label_to_index[label] for label in nuclear_labels_at_pixels])

    # # Build overlap matrix using sparse representation
    # data = np.ones_like(cell_indices, dtype=np.int32)
    # num_cells = len(cell_labels)
    # num_nuclei = len(nuclear_labels)
    # overlap_matrix = coo_matrix(
    #     (data, (cell_indices, nuclear_indices)),
    #     shape=(num_cells, num_nuclei)
    # ).tocsr()

    cell_coords = list(map(lambda x: np.array(x).T, cell_coords))
    nucleus_coords = list(map(lambda x: np.array(x).T, nucleus_coords))
    cell_membrane_coords = list(map(lambda x: np.array(x).T, cell_membrane_coords))

    cell_matched_index_list = []
    nucleus_matched_index_list = []
    cell_matched_list = []
    nucleus_matched_list = []

    for i in range(len(cell_coords)):
        if len(cell_coords[i]) != 0:
            current_cell_coords = cell_coords[i]
            nuclear_search_num = np.unique(
                list(map(lambda x: nuclear_mask[tuple(x)], current_cell_coords))
            )
            best_mismatch_fraction = 1
            whole_cell_best = []
            for j in nuclear_search_num:
                if j != 0:
                    if (j - 1 not in nucleus_matched_index_list) and (
                        i not in cell_matched_index_list
                    ):
                        whole_cell, nucleus, mismatch_fraction = get_matched_cells(
                            cell_coords[i],
                            cell_membrane_coords[i],
                            nucleus_coords[j - 1],
                            mismatch_repair=do_mismatch_repair,
                        )
                        if type(whole_cell) != bool:
                            if mismatch_fraction < best_mismatch_fraction:
                                best_mismatch_fraction = mismatch_fraction
                                whole_cell_best = whole_cell
                                nucleus_best = nucleus
                                i_ind = i
                                j_ind = j - 1
            if len(whole_cell_best) > 0:
                cell_matched_list.append(whole_cell_best)
                nucleus_matched_list.append(nucleus_best)
                cell_matched_index_list.append(i_ind)
                nucleus_matched_index_list.append(j_ind)

    del cell_coords
    del nucleus_coords

    cell_matched_mask = get_mask(cell_matched_list, whole_cell_mask.shape)
    nuclear_matched_mask = get_mask(nucleus_matched_list, whole_cell_mask.shape)
    cell_membrane_mask = get_boundary(cell_matched_mask)
    nuclear_membrane_mask = get_boundary(nuclear_matched_mask)

    if do_mismatch_repair:
        fraction_matched_cells = 1.0
    else:
        fraction_matched_cells = get_fraction_matched_cells(
            whole_cell_mask, nuclear_mask, cell_matched_mask
        )

    matched_mask_stack[0, :, :] = cell_matched_mask
    matched_mask_stack[1, :, :] = nuclear_matched_mask
    matched_mask_stack[2, :, :] = cell_membrane_mask
    matched_mask_stack[3, :, :] = nuclear_membrane_mask
    return matched_mask_stack, fraction_matched_cells
