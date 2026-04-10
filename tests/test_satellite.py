import unittest
import numpy as np

from scar import fill_bdom_grid

class TestTransform(unittest.TestCase):

    def test_fill_bdom_grid(self):
        
        original_grid = np.random.rand(200, 200)
        reference_grid = original_grid.copy()

        nan_positions = [(2, 3), (5, 5), (7, 1)]
        for y, x in nan_positions:
            original_grid[y, x] = np.nan

        filled_grid = fill_bdom_grid(original_grid)

        self.assertTrue(not np.isnan(filled_grid).any(), \
            "Grid still contains NaNs after interpolation")

        for y in range(original_grid.shape[0]):
            for x in range(original_grid.shape[1]):
                
                if (y, x) not in nan_positions:
                    
                    orig_val = reference_grid[y, x]
                    filled_val = filled_grid[y, x]
                    
                    self.assertTrue(
                        np.isclose(orig_val, filled_val, atol=1e-8),
                        f"Value at ({y}, {x}) changed from {orig_val} to {filled_val}"
                    )

if __name__ == "__main__":
    unittest.main()
