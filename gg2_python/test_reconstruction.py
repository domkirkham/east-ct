import unittest
import numpy as np
from matplotlib import pyplot as plt

from ramp_filter import ramp_filter


class TestRamLak(unittest.TestCase):
    def test_filter_values(self):
        """Checks filter symmetry and minimum value"""
        k = 7
        test_fft = np.ones((2**k, 2**k))
        filter_test = ramp_filter(test_fft, 0.1, 2**k)
        self.assertSequenceEqual(list(filter_test[0]), list(filter_test[2**k-1]))

if __name__ == '__main__':
    unittest.main()