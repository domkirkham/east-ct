import context
import matplotlib.pyplot as plt
from ct_phantom import *
from material_data import MaterialData
import pytest


@pytest.fixture
def material():
    """Return a new names instance."""
    return MaterialData()


@pytest.mark.parametrize("phantom_type, res", [
    (1, 256),
    (2, 256),
    (3, 64),
    (4, 64),
    (5, 16),
    (6, 16),
    (7, 16),
])
def test_ct_phantom(material, phantom_type, res):
    phantom = ct_phantom(material.names, res, phantom_type, 'Titanium')
