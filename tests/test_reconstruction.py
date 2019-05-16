import unittest
import numpy as np
from matplotlib import pyplot as plt
import pytest

from ramp_filter import ramp_filter


def test_filter_values():
    """Checks filter symmetry and minimum value"""
    k = 7
    test_fft = np.ones((2**k, 2**k))
    filter_test = ramp_filter(test_fft, 0.1, 2**k)
    for i in range(2**(k-1)):
        assert np.all(filter_test[i] == filter_test[-(i + 1)])
