import numpy as np
import scipy
from scipy import interpolate
from ct_lib import *
from ct_detect import *


def ct_calibrate(photons_source, material_data: MaterialData, sinogram, scale: float, correct=True):
    """
    ct_calibrate converts CT detections to linearised attenuation.

    sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
    in phantom X and returns a linear attenuation sinogram of shape
    [angles, samples].

    linear attenuation coefficients and energies in mev, and scale is the size of each pixel in x, in cm.

    :param photons_source: is an array - the source energy distribution
    :param material_data: Material coeff. data
    :param sinogram: array of shape [angles, n_samples]
    :param scale: The width between collimators (sapling interval) in cm
    :param correct: ???
    :return: calibrated attenuation sinogram of shape [angles, n_samples]
    """

    # Get dimensions and work out detection for just air of twice the side
    # length (has to be the same as in ct_scan.m)
    n_samples = sinogram.shape[1]

    # Width of the scanned area is twice the width of the phantom:
    max_width = 2 * n_samples * scale  # Width of the scanned area
    # The calibration attenuation is same at every point -> just use a single scan
    calibration_scan = ct_detect(photons_source, material_data.coeff("Air"), max_width)

    # Perform calibration
    total_attenuation = -np.log(sinogram / calibration_scan)

    return total_attenuation
