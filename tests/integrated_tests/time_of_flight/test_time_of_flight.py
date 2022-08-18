from pathlib import Path
import pytest
import numpy as np

from ..testing_utils import read_vtu, l_inf_norm, l1_norm, l2_norm
from .fedm_tof import main as fedm_tof_main

test_dir = Path(__file__).parent
ref_dir = test_dir / "20220707_results"


def get_relative_error(path: Path) -> float:
    with open(path, "r") as f_error:
        return float(f_error.readline().split("=")[-1].strip())


@pytest.fixture
def ref_data():
    return read_vtu(ref_dir / "electrons000000.vtu", field_name="f_3199")


@pytest.fixture
def ref_error():
    return get_relative_error(ref_dir / "relative error.log")


@pytest.fixture(scope="module")
def data_dir(tmpdir_factory):
    tmp_dir = Path(tmpdir_factory.mktemp("fedm_tof"))
    fedm_tof_main(output_dir=tmp_dir)
    return tmp_dir


@pytest.fixture
def tof_data(data_dir):
    path = data_dir / "number density" / "electrons" / "electrons000000.vtu"
    return read_vtu(path, field_name="f_3199")


@pytest.fixture
def tof_error(data_dir):
    return get_relative_error(data_dir / "relative error.log")


def test_time_of_flight_relative_error(ref_error, tof_error):
    # Ensure same errors
    assert np.isclose(tof_error, ref_error)


def test_time_of_flight_electron_number_density(ref_data, tof_data):
    # Get relative error
    error = (tof_data - ref_data) / ref_data
    # Ensure L1, L2, Linf errors all reasonable
    assert l1_norm(error) < 1e-5
    assert l2_norm(error) < 1e-5
    assert l_inf_norm(error) < 1e-3
