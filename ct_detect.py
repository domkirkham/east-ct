import numpy as np
from photons import simulate_photons
from photonsource_data import PhotonSourceData
from material_data import MaterialData


def ct_detect(photon_source: np.ndarray, coeffs, depth, mas=10000) -> np.ndarray:
    """
    ct_detect returns detector photons for given material depths.
    y = ct_detect(p, coeffs, depth, mas) takes a source energy
    distribution photons (energies), a set of material linear attenuation
    coefficients coeffs (materials, energies), and a set of material depths
    in depth (materials, samples) and returns the detections at each sample
    in y (samples).

    mas defines the current-time-product which affects the noise distribution
    for the linear attenuation
    """

    # check photon_source for number of energies
    if not isinstance(photon_source, np.ndarray):
        photon_source = np.array([photon_source])
    if photon_source.ndim > 1:
        raise ValueError('input p has more than one dimension')
    n_energies = len(photon_source)

    # check coeffs is of form (n_materials, energies)
    if not isinstance(coeffs, np.ndarray):
        coeffs = np.array([coeffs]).reshape((1, 1))
    elif coeffs.ndim == 1:
        coeffs = coeffs.reshape((1, len(coeffs)))
    elif coeffs.ndim != 2:
        raise ValueError('input coeffs has more than two dimensions')
    if coeffs.shape[1] != n_energies:
        raise ValueError('input coeffs has different number of energies to input p')
    n_materials = coeffs.shape[0]

    # check depth is of shape [n_materials, n_samples]
    if not isinstance(depth, np.ndarray):
        depth = np.array([depth]).reshape((1, 1))
    elif depth.ndim == 1:
        if n_materials is 1:
            depth = depth.reshape(1, len(depth))
        else:
            depth = depth.reshape(len(depth), 1)
    elif depth.ndim != 2:
        raise ValueError('input depth has more than two dimensions')
    if depth.shape[0] != n_materials:
        raise ValueError('input depth has different number of materials to input coeffs')
    n_samples = depth.shape[1]

    # extend source photon array so it covers all samples
    detector_photons = np.repeat(photon_source[:, None], n_samples, axis=1)  # Shape [n_energies, n_samples]

    # calculate array of residual mev x samples for each material in turn
    for m in range(n_materials):
        detector_photons = simulate_photons(detector_photons, coeffs[m], depth[m])

    # sum this over energies
    detector_photons = np.sum(detector_photons, axis=0)

    # todo: model noise

    # minimum detection is one photon
    detector_photons = np.clip(detector_photons, 1., None)

    return detector_photons
