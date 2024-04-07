# main.py

'''
This script is used to compare the execution time of two different versions of the function `inputs_to_dataset`.
plot the execution time comparison across different channel sizes.

Aurthor:  Saiying(Tina) Ge
Date: April,2024
'''

import numpy as np
import time
import matplotlib.pyplot as plt
from dataset_extract import inputs_to_dataset_v1, inputs_to_dataset_v2


def create_large_image_with_bad_pixels(width, height, channels, bad_pixel_ratio):
    image = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    #change to np.float32
    image = image.astype(np.float32)
    mask = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    # mask = np.zeros((height, width), dtype=np.uint8)
    num_bad_pixels = int(width * height * bad_pixel_ratio)
    for _ in range(num_bad_pixels):
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        # pick a random channel
        c = np.random.randint(0, channels)
        image[y, x, c ] = -1  # Assign a bad pixel value in the image
    return image, mask


def test_performance_over_channels(channel_range, width, height, bad_pixel_ratio, stride_length, patch_size):
    v1_times, v2_times = [], []
    # Test the performance for different channel sizes
    for channels in channel_range:
        image, mask = create_large_image_with_bad_pixels(width, height, channels, bad_pixel_ratio)

        # Measure execution time for v1
        start_v1 = time.time()
        inputs_to_dataset_v1(image, mask, patch_size, stride_length)
        v1_times.append(time.time() - start_v1)

        # Measure execution time for v2
        start_v2 = time.time()
        inputs_to_dataset_v2(image, mask, patch_size, stride_length)
        v2_times.append(time.time() - start_v2)

    return v1_times, v2_times


def plot_execution_times(channel_range, v1_times, v2_times, width, height, bad_pixel_ratio, patch_size, stride_length):
    plt.figure(figsize=(10, 6))
    plt.plot(channel_range, v1_times, '-o', label='Version 1')
    plt.plot(channel_range, v2_times, '-x', label='Version 2')
    plt.xlabel('Number of Channels')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison Across Channel Sizes')
    plt.legend()
    plt.grid(True)

    # Add text box with image and patch information
    plt.figtext(0.75, 0.75,
                f"Image dimensions: {width}x{height}\n"
                f"Bad pixel ratio: {bad_pixel_ratio}\n"
                f"Patch size: {patch_size}\n"
                f"Stride length: {stride_length}",
                ha="center", va="center",
                bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=0.5"))
    # Save the figure to a file
    plt.savefig('performance_comparison.png', bbox_inches='tight', dpi=300)

    plt.show()


def main():
    channel_range = range(3, 100)  # From 1 to 50 channels
    width, height = 1024,768 # Image dimensions
    bad_pixel_ratio = 0.01  # Bad pixel ratio
    stride_length = 2
    patch_size = 5

    v1_times, v2_times = test_performance_over_channels(channel_range, width, height, bad_pixel_ratio, stride_length,
                                                        patch_size)
    plot_execution_times(channel_range, v1_times, v2_times, width, height, bad_pixel_ratio, patch_size, stride_length)


if __name__ == '__main__':
    main()
