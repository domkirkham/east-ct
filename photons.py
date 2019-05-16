import numpy as np
import math
from typing import List, Optional, Union


def simulate_photons(original_energy, coeff, depth: Union[float, np.ndarray]) -> np.ndarray:
    """
    Calculates residual energy for a particular material and depth
    photons(original_energy, coeff, depth, mas) takes the original_energy
    (energy, samples) and works out the residual_energy (energy, samples)
    for a particular material with linear attenuation coefficients given
    by coeff (n_energies), and a set of depths given by depth (samples)

    It is more efficient to calculate this for a range of samples rather then
    one at a time.

    :param original_energy: An array of energies of input photons of shape [n_energies, n_samples]
    :param coeff: An array of material attenuation coeff. of shape [n_energies]
    :param depth: A float or an array of depths to calculate attenuation over with shape [n_samples]
    :return: An array of attenuated (residual) energies for each depth; shape [n_energies, n_samples]
    """

    # Ensure energy has shape [n_energies, n_samples]
    if not isinstance(original_energy, np.ndarray):
        original_energy = np.array([original_energy]).reshape((1, 1))
    elif original_energy.ndim == 1:
        original_energy = original_energy.reshape((len(original_energy), 1))
    elif original_energy.ndim != 2:
        raise ValueError('input original_energy has more than two dimensions')
    n_energies = original_energy.shape[0]
    n_samples = original_energy.shape[1]

    # check coeff is vector with shape [n_energies]
    if not isinstance(coeff, np.ndarray):
        coeff = np.array([coeff])
    elif coeff.ndim != 1:
        raise ValueError('input coeffs has more than one dimension')
    if len(coeff) != n_energies:
        raise ValueError('input coeff has different number of n_energies to input original_energy')

    # check depth is vector of samples
    # depth has shape [n_samples]
    if not isinstance(depth, np.ndarray):
        depth = np.array([depth])
    elif depth.ndim != 1:
        raise ValueError('input depth has more than one dimension')
    if len(depth) != n_samples:
        raise ValueError('input depth has different number of samples to input original_energy')

    # Work out residual energy for each depth and at each energy
    residual_energy = original_energy * np.exp(-np.outer(coeff, depth))

    return residual_energy
