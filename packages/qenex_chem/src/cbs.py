"""
Complete Basis Set (CBS) Extrapolation
========================================
Estimates the complete basis set limit from finite-basis calculations
using the Dunning cc-pVXZ hierarchy.

Theory:
  HF energy converges exponentially with cardinal number X:
    E_HF(X) = E_HF(CBS) + A * exp(-alpha * X)

  Correlation energy converges as inverse cube:
    E_corr(X) = E_corr(CBS) + B * X^(-3)

  These are well-established results:
    - Feller, J. Chem. Phys. 96, 6104 (1992) [HF exponential]
    - Helgaker et al., J. Chem. Phys. 106, 9639 (1997) [corr X^-3]

Cardinal numbers:
  cc-pVDZ: X=2, cc-pVTZ: X=3, cc-pVQZ: X=4, cc-pV5Z: X=5

Usage:
    from cbs import extrapolate_cbs_2point

    result = extrapolate_cbs_2point(
        E_hf_dz=-76.027, E_corr_dz=-0.208,
        E_hf_tz=-76.058, E_corr_tz=-0.279,
    )
    print(f"CBS total: {result['E_total_cbs']:.6f} Eh")
"""

import math

__all__ = [
    "extrapolate_cbs_2point",
    "extrapolate_hf_2point",
    "extrapolate_corr_2point",
    "CARDINAL_NUMBERS",
]

# Cardinal numbers for cc-pVXZ basis sets
CARDINAL_NUMBERS = {
    "cc-pvdz": 2,
    "cc-pvtz": 3,
    "cc-pvqz": 4,
    "cc-pv5z": 5,
    "dz": 2,
    "tz": 3,
    "qz": 4,
    "5z": 5,
}

# Feller (1992) optimal alpha for HF exponential extrapolation
HF_ALPHA = 1.63


def extrapolate_hf_2point(E_hf_X, E_hf_Y, X=2, Y=3, alpha=HF_ALPHA):
    """
    Extrapolate HF energy to CBS limit using 2-point exponential formula.

    E_HF(n) = E_HF(CBS) + A * exp(-alpha * n)

    Solving for E_HF(CBS) from two points (X, Y):
      E_HF(CBS) = [E_X * exp(-alpha*Y) - E_Y * exp(-alpha*X)]
                  / [exp(-alpha*Y) - exp(-alpha*X)]

    Args:
        E_hf_X: HF energy with smaller basis (cardinal number X)
        E_hf_Y: HF energy with larger basis (cardinal number Y)
        X: Cardinal number of smaller basis (default: 2 for DZ)
        Y: Cardinal number of larger basis (default: 3 for TZ)
        alpha: Exponential parameter (default: 1.63, Feller 1992)

    Returns:
        E_HF(CBS) in Hartree
    """
    exp_X = math.exp(-alpha * X)
    exp_Y = math.exp(-alpha * Y)
    return (E_hf_X * exp_Y - E_hf_Y * exp_X) / (exp_Y - exp_X)


def extrapolate_corr_2point(E_corr_X, E_corr_Y, X=2, Y=3):
    """
    Extrapolate correlation energy to CBS limit using Helgaker X^-3 formula.

    E_corr(n) = E_corr(CBS) + B * n^(-3)

    Solving for E_corr(CBS) from two points (X, Y):
      E_corr(CBS) = [X^3 * E_corr(X) - Y^3 * E_corr(Y)] / [X^3 - Y^3]

    Args:
        E_corr_X: Correlation energy with smaller basis (cardinal X)
        E_corr_Y: Correlation energy with larger basis (cardinal Y)
        X: Cardinal number of smaller basis (default: 2 for DZ)
        Y: Cardinal number of larger basis (default: 3 for TZ)

    Returns:
        E_corr(CBS) in Hartree

    Reference:
        Helgaker, Klopper, Koch, Noga, J. Chem. Phys. 106, 9639 (1997)
    """
    X3 = X**3
    Y3 = Y**3
    return (X3 * E_corr_X - Y3 * E_corr_Y) / (X3 - Y3)


def extrapolate_cbs_2point(
    E_hf_dz,
    E_corr_dz,
    E_hf_tz,
    E_corr_tz,
    X=2,
    Y=3,
    alpha=HF_ALPHA,
):
    """
    Full CBS extrapolation from DZ/TZ pair.

    Separately extrapolates HF (exponential) and correlation (X^-3)
    components to the CBS limit, then combines.

    Args:
        E_hf_dz: RHF energy with cc-pVDZ (or smaller basis)
        E_corr_dz: Correlation energy with cc-pVDZ
        E_hf_tz: RHF energy with cc-pVTZ (or larger basis)
        E_corr_tz: Correlation energy with cc-pVTZ

    Returns:
        dict with keys:
            E_hf_cbs: Extrapolated HF energy
            E_corr_cbs: Extrapolated correlation energy
            E_total_cbs: Total CBS energy (HF + corr)
            E_total_dz: Total DZ energy (for comparison)
            E_total_tz: Total TZ energy (for comparison)
            delta_dz: CBS correction from DZ (E_cbs - E_dz)
            delta_tz: CBS correction from TZ (E_cbs - E_tz)
    """
    E_hf_cbs = extrapolate_hf_2point(E_hf_dz, E_hf_tz, X, Y, alpha)
    E_corr_cbs = extrapolate_corr_2point(E_corr_dz, E_corr_tz, X, Y)
    E_total_cbs = E_hf_cbs + E_corr_cbs

    E_total_dz = E_hf_dz + E_corr_dz
    E_total_tz = E_hf_tz + E_corr_tz

    return {
        "E_hf_cbs": E_hf_cbs,
        "E_corr_cbs": E_corr_cbs,
        "E_total_cbs": E_total_cbs,
        "E_total_dz": E_total_dz,
        "E_total_tz": E_total_tz,
        "delta_dz": E_total_cbs - E_total_dz,
        "delta_tz": E_total_cbs - E_total_tz,
    }


def compute_cbs(molecule, verbose=True):
    """
    Compute CBS-extrapolated MP2 energy for a molecule.

    Runs RHF + MP2 with both cc-pVDZ and cc-pVTZ, then extrapolates
    to the complete basis set limit.

    Args:
        molecule: Molecule object
        verbose: Print results

    Returns:
        dict with CBS results (same as extrapolate_cbs_2point)
    """
    from molecule import Molecule
    from solver import HartreeFockSolver, MP2Solver

    # Clone molecule with different basis sets
    mol_dz = Molecule(
        molecule.atoms, molecule.charge, molecule.multiplicity, basis_name="cc-pvdz"
    )
    mol_tz = Molecule(
        molecule.atoms, molecule.charge, molecule.multiplicity, basis_name="cc-pvtz"
    )

    # cc-pVDZ
    hf_dz = HartreeFockSolver()
    E_hf_dz, _ = hf_dz.compute_energy(mol_dz, verbose=False)
    mp2_dz = MP2Solver()
    _, E_corr_dz = mp2_dz.compute_correlation(hf_dz, mol_dz, verbose=False)

    # cc-pVTZ
    hf_tz = HartreeFockSolver()
    E_hf_tz, _ = hf_tz.compute_energy(mol_tz, verbose=False)
    mp2_tz = MP2Solver()
    _, E_corr_tz = mp2_tz.compute_correlation(hf_tz, mol_tz, verbose=False)

    result = extrapolate_cbs_2point(E_hf_dz, E_corr_dz, E_hf_tz, E_corr_tz)

    if verbose:
        print("=" * 60)
        print("CBS Extrapolation (DZ/TZ 2-point)")
        print("=" * 60)
        print(f"  cc-pVDZ:  E_HF = {E_hf_dz:16.10f}  E_corr = {E_corr_dz:12.10f}")
        print(f"  cc-pVTZ:  E_HF = {E_hf_tz:16.10f}  E_corr = {E_corr_tz:12.10f}")
        print(
            f"  CBS:      E_HF = {result['E_hf_cbs']:16.10f}  E_corr = {result['E_corr_cbs']:12.10f}"
        )
        print(f"")
        print(f"  E_total(DZ)  = {result['E_total_dz']:16.10f} Eh")
        print(f"  E_total(TZ)  = {result['E_total_tz']:16.10f} Eh")
        print(f"  E_total(CBS) = {result['E_total_cbs']:16.10f} Eh")
        print(f"")
        print(
            f"  CBS correction from DZ: {result['delta_dz']:+.6f} Eh ({result['delta_dz'] * 627.5094:.2f} kcal/mol)"
        )
        print(
            f"  CBS correction from TZ: {result['delta_tz']:+.6f} Eh ({result['delta_tz'] * 627.5094:.2f} kcal/mol)"
        )
        print("=" * 60)

    return result
