import numpy as np
import scipy
from scipy import interpolate
from ct_lib import *
from ct_detect import *


def ct_calibrate(photons_source, material_data: MaterialData, sinogram, scale: float, correct=True, order=4, material="Water"):
    """
    ct_calibrate converts CT detections to linearised attenuation.

    sinogram = ct_calibrate(photons, material, sinogram, scale) takes the CT detection sinogram
    in phantom X and returns a linear attenuation sinogram of shape
    [angles, samples].

    linear attenuation coefficients and energies in mev, and scale is the size of each pixel in x, in cm.

    :param photons_source: is an array - the source energy distribution
    :param material_data: Material coeff. data
    :param sinogram: array of shape [angles, n_samples]
    :param scale: The width between collimators (sampling interval) in cm
    :param correct: Whether to include beam hardening correction
    :param order: order of polyfit for correction
    :param material: the material to correct beam hardening for
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

    if correct:

        # Create list of depths for fitting polynomial
        depths = np.arange(0, 0.5*sinogram.shape[1], 0.1)
        depths = depths*scale

        # Get the coefficients for the polynomial
        fit_coeffs = get_attenuation(photons_source, material_data, [material], depths, calibration_scan, order)

        # Generate matrices of 1, mu, mu^2 etc.
        atten_matrix = calibrate_attenuation(
            material_data, total_attenuation, order, material_data.names.index(material))

        # Get depths as calculated by polynomial
        equivalent_depths = np.matmul(atten_matrix, np.flip(fit_coeffs[material]))

        # times by C so mu_c = mu_m at low attenuations (divide by linear coeff)
        total_attenuation = equivalent_depths / fit_coeffs[material][-2]

    return total_attenuation


def get_attenuation(photons_source, material_data: MaterialData, materials, depths, calibration, fit_order: int):

    # Calculate attenuation at each depth for each material
    mapped_energies = tuple(map(
        lambda material: ct_detect(photons_source, material_data.coeff(material), depths), materials))

    # Put these in a dictionary with the material names as the key
    energies_dict = {f'{materials[i]}': np.float32(-np.log(vals/calibration)) for i, vals in enumerate(mapped_energies)}

    # Polynomial fit for each material over the depths
    fit_dict_coeffs = {f'{mat}':np.polyfit(energies, depths, fit_order) for mat, energies in energies_dict.items()}

    print(fit_dict_coeffs)

    return fit_dict_coeffs


def calibrate_attenuation(material_data, attenuation, order, ind):

    atten_matrix = attenuation**(np.arange(order+1))[:, None, None]
    atten_matrix = np.swapaxes(np.swapaxes(atten_matrix, 2, 0), 1, 0)

    return atten_matrix

