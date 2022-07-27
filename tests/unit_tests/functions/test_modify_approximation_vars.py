from fedm.functions import modify_approximation_vars
import numpy as np
import pytest


def test_modify_approximation_vars_LFA():
    """
    Expect to remove the first particle.
    """
    n_species = 3
    particles = ["foo", "bar", "baz"]
    masses = [0.5, 2.0, 5.0]
    charges = [-1.5, 0.0, 3.0]
    n_species, n_equations, particles, masses, charges = modify_approximation_vars(
        "LFA", n_species, particles, masses, charges
    )
    assert n_species == 2
    assert n_equations == 3
    assert np.array_equal(particles, ["bar", "baz"])
    assert np.array_equal(masses, [2.0, 5.0])
    assert np.array_equal(charges, [0.0, 3.0])


def test_modify_approximation_vars_LMEA():
    """
    Expect no changes, except that n_equations is determined
    """
    n_species = 3
    particles = ["foo", "bar", "baz"]
    masses = [0.5, 2.0, 5.0]
    charges = [-1.5, 0.0, 3.0]
    n_species, n_equations, particles, masses, charges = modify_approximation_vars(
        "LMEA", n_species, particles, masses, charges
    )
    assert n_species == 3
    assert n_equations == 4
    assert np.array_equal(particles, ["foo", "bar", "baz"])
    assert np.array_equal(masses, [0.5, 2.0, 5.0])
    assert np.array_equal(charges, [-1.5, 0.0, 3.0])


def test_modify_approximation_vars_invalid():
    """
    Expect ValueError raised for incorrect approximation type
    """
    n_species = 3
    particles = ["foo", "bar", "baz"]
    masses = [0.5, 2.0, 5.0]
    charges = [-1.5, 0.0, 3.0]
    with pytest.raises(ValueError) as exc_info:
        n_species, n_equations, particles, masses, charges = modify_approximation_vars(
            "helloworld", n_species, particles, masses, charges
        )
    assert "helloworld" in str(exc_info.value)
