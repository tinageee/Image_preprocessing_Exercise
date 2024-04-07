'''
This script contains two functions to extract patches from an image and its corresponding mask.
The first function, inputs_to_dataset_v1, extracts patches from the image and mask using a nested loop.
The second function, inputs_to_dataset_v2, extracts patches from the image and mask using a vectorized approach.

Aurthor:  Saiying(Tina) Ge
Date: April,2024
'''

import numpy as np

def check_inputs(image, mask, patch_size, stride_length):
    '''
    check the inputs value, format and dimensions
    :param image: the image to be checked
    :param mask: the mask to be checked
    :param patch_size: the size of the patch
    :param stride_length: the length of the stride
    :return:    None
    '''
    if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
        raise TypeError("Image and mask must be numpy arrays.")
    if image.shape[:2] != mask.shape:
        raise ValueError("Image and mask dimensions must match.")
    if patch_size <= 0 or stride_length <= 0:
        raise ValueError("Patch size and stride length must be positive integers.")
    if patch_size > image.shape[0] or patch_size > image.shape[1]:
        raise ValueError("Patch size must be smaller than the image dimensions.")
    if stride_length > patch_size:
        raise ValueError("Stride length must be no greater than the patch size.") # All patches should be fully contained in the image

def inputs_to_dataset_v1(image, mask, patch_size, stride_length):
    '''
    extract the patches from the image and mask.
    1. check the inputs
    2. iterate over the image to extract patches
    3. check if the patch contains -1, if not append the patch to the list and the corresponding mask to the mask list
    4. change the list to numpy array

    :param image: the image
    :param mask: the mask
    :param patch_size:  the size of the patch
    :param stride_length:   the length of the stride
    :return:    patches, mask_patches
    '''
    try:
        # Check inputs
        check_inputs(image, mask, patch_size, stride_length)

        image_patches = []
        mask_patches = []

        # Iterate over the image to extract patches
        # "starting with the top (left to right) and finishing with the bottom (left to right)"
        for i in range(0, image.shape[0] - patch_size + 1, stride_length):
            for j in range(0, image.shape[1] - patch_size + 1, stride_length):
                patch = image[i:i + patch_size, j:j + patch_size, :]
                # Check if the patch contains -1
                if not np.any(patch == -1):
                    # Append the patch to the list
                    image_patches.append(patch)
                    # Append the corresponding mask to the mask list
                    mask_patches.append(mask[i:i + patch_size, j:j + patch_size])

        # Change the list to numpy array
        patches = np.array(image_patches)
        mask_patches = np.array(mask_patches)
        return patches, mask_patches
    except Exception as e:
        print(f"An error occurred in inputs_to_dataset_v1: {e}")
        raise

## V2: vectorized version of the function

def find_valid_patch_starts_vectorized(image, patch_size, stride_length):
    '''
    assume stride length is not greater than the patch size, find the valid patch starts in the image

    1. get the height, width and channels of the image
    2. create a bad_point_mask to check if any channel is -1 for each point,
    3. create a valid_starts_mask to check if the patch contains -1,

    :param image:  the image
    :param patch_size:  the size of the patch
    :param stride_length:   the length of the stride
    :return:    valid starts indices
    '''
    try:
        # Get the height, width, and channels of the image
        height, width, _ = image.shape
        # Find where any channel is -1 for each point
        bad_point_mask = np.any(image == -1, axis=-1)
        # Create a valid_starts_mask to check if the patch contains -1, if not, set the mask to False
        valid_starts_mask = np.ones((height - patch_size + 1, width - patch_size + 1), dtype=bool)
        # Set the mask to False if the patch contains -1,considering the patch size
        for m, n in np.argwhere(bad_point_mask):
            valid_starts_mask[max(0, m - patch_size + 1):min(height - patch_size, m) + 1,
                              max(0, n - patch_size + 1):min(width - patch_size, n) + 1] = False
        # Get the valid starts indices
        valid_starts_indices = np.argwhere(valid_starts_mask)[::stride_length]
        return valid_starts_indices
    except Exception as e:
        print(f"An error occurred in find_valid_patch_starts_vectorized: {e}")
        raise

def extract_patches_from_valid_starts_vectorized(data, valid_starts, patch_size):
    '''
    extract the image patches or mask patches from the valid starts
    :param data:    the original image or masks to be extracted
    :param valid_starts:  the valid starts
    :param patch_size:  the size of the patch
    :return:    patches or masks
    '''
    try:
        patches = []
        for i, j in valid_starts:
            # Extract the image patches from the valid starts
            if data.ndim == 3:
                patch = data[i:i + patch_size, j:j + patch_size, :]
            # Extract the mask patches from the valid starts
            elif data.ndim == 2:
                patch = data[i:i + patch_size, j:j + patch_size]
            else:
                raise ValueError("Data must be either 2D or 3D.")
            patches.append(patch)
        return np.array(patches)
    except Exception as e:
        print(f"An error occurred in extract_patches_from_valid_starts_vectorized: {e}")
        raise

def inputs_to_dataset_v2(image, mask, patch_size, stride_length):
    '''
    extract the patches from the image and mask using the vectorized version of the function
    :param image:   the image
    :param mask:    the mask
    :param patch_size:  the size of the patch
    :param stride_length:  the length of the stride
    :return:    image_patches, mask_patches
    '''
    try:
        # Check inputs
        check_inputs(image, mask, patch_size, stride_length)
        # Find the valid patch starts in the image
        valid_starts = find_valid_patch_starts_vectorized(image, patch_size, stride_length)
        # Extract the image patches and mask patches from the valid starts
        image_patches = extract_patches_from_valid_starts_vectorized(image, valid_starts, patch_size)
        mask_patches = extract_patches_from_valid_starts_vectorized(mask, valid_starts, patch_size)
        return image_patches, mask_patches
    except Exception as e:
        print(f"An error occurred in inputs_to_dataset_v2: {e}")
        raise
