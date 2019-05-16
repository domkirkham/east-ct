import context
import numpy as np
from ct_detect import *
from photons import simulate_photons
from material_data import MaterialData
from photonsource_data import PhotonSourceData


def test_photons_init():
    material = MaterialData()
    source = PhotonSourceData()
    coeff = material.coeff('Bone')
    y = simulate_photons(source.photons[0], coeff, [0.0, 0.1, 200])
    assert isinstance(y, np.ndarray)
    return


def test_photons_with_ct_detect():
    material = MaterialData()
    source = PhotonSourceData()
    coeff = material.coeff('Water')

    y = ct_detect(source.photons[0], coeff, np.arange(0, 10.1, 0.1), 1)

    assert isinstance(y, np.ndarray)
    return
