from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ..testing_utils import read_h5, l_inf_norm, l1_norm, l2_norm
from .fedm_gd import main as fedm_gd_main

test_dir = Path(__file__).parent
input_dir = test_dir / "file_input"
ref_dir = test_dir / "20220707_results"

keys = ["electrons", "Ar_1p0", "Ar_plus", "Ar_star"]


def get_relative_error(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+")


@pytest.fixture
def ref_data():
    return {key: read_h5(ref_dir / (key + ".h5"), key=key) for key in keys}


@pytest.fixture
def ref_error():
    return get_relative_error(ref_dir / "relative error.log")


@pytest.fixture(scope="module")
def data_dir(tmpdir_factory):
    tmp_dir = Path(tmpdir_factory.mktemp("fedm_gd"))
    fedm_gd_main(input_dir=input_dir, output_dir=tmp_dir)
    return tmp_dir


@pytest.fixture
def gd_data(data_dir):
    paths = [data_dir / "number density" / key / (key + ".h5") for key in keys]
    return {key: read_h5(path, key=key) for path, key in zip(paths, keys)}


@pytest.fixture
def gd_error(data_dir):
    return get_relative_error(data_dir / "relative error.log")


def test_glow_discharge_relative_error(ref_error, gd_error):
    # Ensure same errors
    assert np.allclose(gd_error, ref_error)


@pytest.mark.parametrize("particle", keys)
def test_glow_discharge_number_density(ref_data, gd_data, particle):
    # Get relative error
    gd_list, ref_list = gd_data[particle], ref_data[particle]
    for gd, ref in zip(gd_list, ref_list):
        error = (gd - ref) / ref
        # Ensure L1, L2, Linf errors all reasonable
        assert l1_norm(error) < 1e-5
        assert l2_norm(error) < 1e-5
        assert l_inf_norm(error) < 1e-3
