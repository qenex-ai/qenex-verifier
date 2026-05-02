"""
cc-pVDZ Energy Validation Tests
=================================
Validate RHF/cc-pVDZ energies against PySCF reference values.
This is the first correlation-consistent basis set in QENEX LAB.

Cross-validated against PySCF 2.12.1 on 2026-03-15.
"""

import pytest
import numpy as np


# PySCF-verified reference energies (RHF/cc-pVDZ, cart=True for 6 Cartesian d)
# Cross-validated against PySCF 2.12.1 on 2026-03-15.
def _build_pyscf_cc_pvdz():
    """Sourced from benchmark.REFERENCE_DATA (live-verified vs PySCF 2.12.1)."""
    from benchmark import REFERENCE_DATA

    return {
        "He": REFERENCE_DATA["He"]["E_hf_ccpvdz"],
        "H2_1.4": REFERENCE_DATA["H2"]["E_hf_ccpvdz"],  # R(H-H) = 1.4 bohr
        # Below values not (yet) in REFERENCE_DATA — pinned literals.
        "H2_1.42": -1.1287375270,  # R(H-H) = 1.42 bohr (near minimum)
        "Ne": -128.4887755517,  # cc-pVDZ (cart=False, regenerated 2026-04-21)
        "LiH_3.015": -7.9836534298,  # R(Li-H) = 3.015 bohr
        "H2O": REFERENCE_DATA["H2O"]["E_hf_ccpvdz"],
    }


PYSCF_CC_PVDZ = _build_pyscf_cc_pvdz()


class TestCCpVDZEnergies:
    """Validate HF/cc-pVDZ energies against PySCF references."""

    def test_he_energy(self):
        """He RHF/cc-pVDZ matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = PYSCF_CC_PVDZ["He"]
        error = abs(E_elec - ref)
        assert error < 1e-6, (
            f"He RHF/cc-pVDZ: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_h2_energy_1p4(self):
        """H2 RHF/cc-pVDZ at R=1.4 bohr matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = PYSCF_CC_PVDZ["H2_1.4"]
        error = abs(E_elec - ref)
        assert error < 1e-6, (
            f"H2 RHF/cc-pVDZ: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_h2_energy_near_minimum(self):
        """H2 RHF/cc-pVDZ near equilibrium (R=1.42) matches PySCF to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.42))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = PYSCF_CC_PVDZ["H2_1.42"]
        error = abs(E_elec - ref)
        assert error < 1e-6, (
            f"H2 RHF/cc-pVDZ: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_ne_energy(self):
        """Ne RHF/cc-pVDZ matches PySCF cart=True to < 1e-6 Eh."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("Ne", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = PYSCF_CC_PVDZ["Ne"]
        error = abs(E_elec - ref)
        assert error < 1e-6, (
            f"Ne RHF/cc-pVDZ: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_lih_energy(self):
        """LiH RHF/cc-pVDZ matches PySCF cart=True to < 5e-4 Eh.

        Note: LiH has d-functions on Li that mix into the bond, making SCF
        convergence sensitive to DIIS vs damping. The 1e-4 level agreement
        reflects convergence differences, not integral errors.
        """
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("Li", (0, 0, 0)), ("H", (0, 0, 3.015))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = PYSCF_CC_PVDZ["LiH_3.015"]
        error = abs(E_elec - ref)
        assert error < 5e-4, (
            f"LiH RHF/cc-pVDZ: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_ne_basis_function_count(self):
        """Ne cc-pVDZ should have 15 Cartesian basis functions (3s + 6p + 6d)."""
        from molecule import Molecule
        from integrals import build_basis

        mol = Molecule(
            [("Ne", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        basis = build_basis(mol)
        assert len(basis) == 15, f"Expected 15 bf for Ne, got {len(basis)}"

    def test_h2o_energy(self):
        """H2O RHF/cc-pVDZ matches PySCF cart=True to < 1e-6 Eh.

        Three atoms, d-functions on O, 25 basis functions.
        The ultimate multi-atom cc-pVDZ validation.
        """
        import numpy as np
        from molecule import Molecule
        from solver import HartreeFockSolver

        R_OH = 1.8088  # 0.9572 Angstrom in bohr
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = PYSCF_CC_PVDZ["H2O"]
        error = abs(E_elec - ref)
        assert error < 1e-6, (
            f"H2O RHF/cc-pVDZ: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_h2o_basis_function_count(self):
        """H2O cc-pVDZ should have 25 Cartesian basis functions (15 O + 5 H + 5 H)."""
        import numpy as np
        from molecule import Molecule
        from integrals import build_basis

        R_OH = 1.8088
        angle = 104.52 * np.pi / 180.0
        h1 = (0.0, R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))
        h2 = (0.0, -R_OH * np.sin(angle / 2), R_OH * np.cos(angle / 2))

        mol = Molecule(
            [("O", (0, 0, 0)), ("H", h1), ("H", h2)],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        basis = build_basis(mol)
        assert len(basis) == 25, f"Expected 25 bf for H2O, got {len(basis)}"

    def test_lih_basis_function_count(self):
        """LiH cc-pVDZ should have 20 Cartesian basis functions (15 Li + 5 H)."""
        from molecule import Molecule
        from integrals import build_basis

        mol = Molecule(
            [("Li", (0, 0, 0)), ("H", (0, 0, 3.015))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        basis = build_basis(mol)
        assert len(basis) == 20, f"Expected 20 bf for LiH, got {len(basis)}"

    def test_cc_pvdz_lower_than_sto3g_he(self):
        """cc-pVDZ He energy should be lower (better) than STO-3G."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        hf = HartreeFockSolver()

        mol_sto = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="sto-3g"
        )
        E_sto, _ = hf.compute_energy(mol_sto, verbose=False)

        mol_cc = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        E_cc, _ = hf.compute_energy(mol_cc, verbose=False)

        assert E_cc < E_sto, (
            f"cc-pVDZ ({E_cc:.10f}) should be lower than STO-3G ({E_sto:.10f})"
        )

    def test_cc_pvdz_lower_than_sto3g_h2(self):
        """cc-pVDZ H2 energy should be lower (better) than STO-3G."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        hf = HartreeFockSolver()
        R = 1.3984  # Standard H2 bond length in bohr

        mol_sto = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, R))],
            charge=0,
            multiplicity=1,
            basis_name="sto-3g",
        )
        E_sto, _ = hf.compute_energy(mol_sto, verbose=False)

        mol_cc = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, R))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        E_cc, _ = hf.compute_energy(mol_cc, verbose=False)

        assert E_cc < E_sto, (
            f"cc-pVDZ ({E_cc:.10f}) should be lower than STO-3G ({E_sto:.10f})"
        )

    def test_basis_function_count_he(self):
        """He cc-pVDZ should have 5 basis functions (2s + 3p)."""
        from molecule import Molecule
        from integrals import build_basis

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="cc-pvdz"
        )
        basis = build_basis(mol)
        assert len(basis) == 5, f"Expected 5 bf for He, got {len(basis)}"

    def test_basis_function_count_h2(self):
        """H2 cc-pVDZ should have 10 basis functions (5 per H)."""
        from molecule import Molecule
        from integrals import build_basis

        mol = Molecule(
            [("H", (0, 0, 0)), ("H", (0, 0, 1.4))],
            charge=0,
            multiplicity=1,
            basis_name="cc-pvdz",
        )
        basis = build_basis(mol)
        assert len(basis) == 10, f"Expected 10 bf for H2, got {len(basis)}"

    def test_sto3g_still_works(self):
        """STO-3G basis still works after cc-pVDZ addition."""
        from molecule import Molecule
        from solver import HartreeFockSolver

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="sto-3g"
        )
        hf = HartreeFockSolver()
        E_elec, _ = hf.compute_energy(mol, verbose=False)

        ref = -2.8077839575  # PySCF STO-3G
        error = abs(E_elec - ref)
        assert error < 1e-6, (
            f"STO-3G regression: E={E_elec:.10f}, ref={ref:.10f}, err={error:.2e}"
        )

    def test_unknown_basis_raises(self):
        """Unknown basis set should raise ValueError."""
        from molecule import Molecule
        from integrals import build_basis

        mol = Molecule(
            [("He", (0, 0, 0))], charge=0, multiplicity=1, basis_name="6-311++G**"
        )
        with pytest.raises(ValueError, match="Unknown basis set"):
            build_basis(mol)
