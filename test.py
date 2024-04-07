import unittest
import numpy as np

from dataset_extract import inputs_to_dataset_v1,inputs_to_dataset_v2

class TestInputsToDataset(unittest.TestCase):
    def given_example(self):
        # image with height 3, width 3, and 3 channels.
        image = np.array([
            [[1, 2, 3], [4, 5, 6], [-1, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
        ], dtype=np.float32)

        # mask corresponding to above image
        mask = np.array(
            [[1, 4, 7],
             [10, 4, 16],
             [19, 22, 255]],
            dtype=np.uint8)

        stride_length = 1
        patch_size = 2

        image_patches, mask_patches = inputs_to_dataset_v2(image, mask, patch_size, stride_length)

        out_put_image_patches = np.array(
            [[[[1, 2, 3], [4, 5, 6]],
              [[10, 11, 12], [13, 14, 15]]],
             [[[4, 5, 6], [-1, 8, 9]],
              [[13, 14, 15], [16, 17, 18]]],
             [[[10, 11, 12], [13, 14, 15]],
              [[19, 20, 21], [22, 23, 24]]]], dtype=np.float32)

        self.assertTrue(np.allclose(image_patches, out_put_image_patches))

        out_put_mask_patches = np.array(
            [[[1, 4],
              [10, 4]],
             [[10, 4],
              [19, 22]],
             [[4, 16],
              [22, 255]]], dtype=np.uint8)

        self.assertTrue(np.allclose(mask_patches, out_put_mask_patches))

if __name__ == '__main__':
    unittest.main()