import pytest
import numpy as np
import integrals as ints
from solver import HartreeFockSolver
from molecule import Molecule


def test_p_orbital_eri():
    """Verify that ERI involving p-orbitals are non-zero and correct."""
    # (px px | px px)
    bf_x = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (1, 0, 0))
    val_xxxx = ints.eri(bf_x, bf_x, bf_x, bf_x)
    assert abs(val_xxxx - 0.921510) < 1e-5

    # (px px | py py)
    bf_y = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 1, 0))
    val_xxyy = ints.eri(bf_x, bf_x, bf_y, bf_y)
    assert abs(val_xxyy - 0.808672) < 1e-5

    # (px px | pz pz)
    bf_z = ints.BasisFunction([0.0, 0.0, 0.0], 1.0, 1.0, (0, 0, 1))
    val_xxzz = ints.eri(bf_x, bf_x, bf_z, bf_z)
    assert abs(val_xxzz - 0.808672) < 1e-5

    # (px py | px py)
    val_xyxy = ints.eri(bf_x, bf_y, bf_x, bf_y)
    assert abs(val_xyxy - 0.056419) < 1e-5


def test_water_sto3g_energy():
    """Verify that Water STO-3G energy matches reference."""
    # Standard geometry for Water STO-3G test
    mol = Molecule(
        [("O", (0.0, 0.0, 0.0)), ("H", (1.809, 0.0, 0.0)), ("H", (-0.419, 1.760, 0.0))]
    )

    solver = HartreeFockSolver()
    # API returns (total_energy, electronic_energy)
    E_tot, E_elec = solver.compute_energy(mol, verbose=False)

    # Reference: -74.963 Ha (STO-3G water total energy)
    assert abs(E_tot - (-74.963)) < 0.01


if __name__ == "__main__":
    # Manual run
    try:
        test_p_orbital_eri()
        print("P-Orbital ERI Test: PASS")
    except AssertionError as e:
        print(f"P-Orbital ERI Test: FAIL {e}")

    try:
        test_water_sto3g_energy()
        print("Water Energy Test: PASS")
    except AssertionError as e:
        print(f"Water Energy Test: FAIL {e}")
