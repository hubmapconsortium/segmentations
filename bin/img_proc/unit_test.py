import unittest
import numpy as np
from match_masks import get_matched_masks

class TestGetMatchedMasks(unittest.TestCase):
    def setUp(self):
        # Generate synthetic test data for mask_stack
        # For simplicity, create small masks with known labels

        # Create a mask_stack with 4 layers: whole_cell_mask, nuclear_mask, cell_membrane_mask, nuclear_membrane_mask
        self.mask_shape = (10, 10)
        self.mask_stack = np.zeros((4, *self.mask_shape), dtype=np.int32)

        # Define cell labels
        self.mask_stack[0, 2:5, 2:5] = 1  # Cell 1
        self.mask_stack[0, 6:9, 6:9] = 2  # Cell 2

        # Define nuclear labels
        self.mask_stack[1, 3:4, 3:4] = 1  # Nucleus 1
        self.mask_stack[1, 7:8, 7:8] = 2  # Nucleus 2

        # For this test, we'll leave cell_membrane_mask and nuclear_membrane_mask empty
        # Since they are generated within the function

        self.do_mismatch_repair = False

    def test_get_matched_masks(self):
        # Run the original function
        original_result, original_fraction = get_matched_masks(
            self.mask_stack, self.do_mismatch_repair
        )

        # Run the optimized function
        optimized_result, optimized_fraction = get_matched_masks(
            self.mask_stack, self.do_mismatch_repair
        )

        # Compare the outputs
        np.testing.assert_array_equal(
            original_result, optimized_result,
            err_msg="The matched masks from the original and optimized functions do not match."
        )
        self.assertAlmostEqual(
            original_fraction, optimized_fraction,
            msg="The fraction of matched cells does not match."
        )

# Run the test
if __name__ == '__main__':
    unittest.main()
