from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from ..testing_utils import read_vtu, l_inf_norm, l1_norm, l2_norm
from .fedm_streamer import main

test_dir = Path(__file__).parent
ref_dir = test_dir / "20220707_results"


def get_relative_error(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+")


@pytest.fixture
def ref_data():
    return {
        "ions": read_vtu(ref_dir / "Ions000000.vtu", field_name="ions"),
        "electrons": read_vtu(ref_dir / "electrons000000.vtu", field_name="electrons"),
    }


@pytest.fixture
def ref_error():
    return get_relative_error(ref_dir / "relative error.log")


@pytest.fixture(scope="module")
def data_dir(tmpdir_factory):
    tmp_dir = Path(tmpdir_factory.mktemp("fedm_streamer"))
    input_dir = Path(__file__).parent / "file_input"
    main(input_dir=input_dir, output_dir=tmp_dir)
    return tmp_dir


@pytest.fixture
def streamer_data(data_dir):
    electron_path = data_dir / "number density" / "electrons" / "electrons000000.vtu"
    ion_path = data_dir / "number density" / "Ions" / "Ions000000.vtu"
    return {
        "electrons": read_vtu(electron_path, field_name="electrons"),
        "ions": read_vtu(ion_path, field_name="ions"),
    }


@pytest.fixture
def streamer_error(data_dir):
    return get_relative_error(data_dir / "relative error.log")


def test_streamer_discharge_relative_error(ref_error, streamer_error):
    # Ensure same errors
    assert np.allclose(streamer_error, ref_error)


@pytest.mark.parametrize("particle", ["electrons", "ions"])
def test_streamer_discharge_number_density(ref_data, streamer_data, particle):
    # Get relative error
    error = (streamer_data[particle] - ref_data[particle]) / ref_data[particle]
    # Ensure L1, L2, Linf errors all reasonable
    assert l1_norm(error) < 1e-5
    assert l2_norm(error) < 1e-5
    assert l_inf_norm(error) < 1e-3
