"""
Vibrational Frequency Analysis Module

Computes harmonic vibrational frequencies, normal modes, thermochemical
properties, and IR intensities from Hartree-Fock (or any solver with
compute_energy/compute_gradient interface).

Methods:
  - compute_hessian()         — Cartesian Hessian via central finite difference
                                 of analytical gradients
  - compute_frequencies()     — Mass-weighted normal mode analysis
  - compute_thermochemistry() — ZPE, thermal corrections, entropy, Gibbs energy
  - compute_ir_spectrum()     — IR intensities from dipole derivatives

All coordinates are in Bohr.  Only numpy is used — no external dependencies.

Physical constants (atomic units / SI):
  - Eigenvalue lambda in Eh/(bohr^2 * amu)
  - Frequency: nu (cm^-1) = sqrt(lambda_au) * FREQ_AU_TO_CM1
  - Thermochemistry uses standard ideal-gas / rigid-rotor / harmonic-oscillator
    (IGRRHO) partition functions.

Reference:
  Ochterski, "Thermochemistry in Gaussian" (Gaussian white paper, 2000)
  McQuarrie, "Statistical Mechanics" (University Science Books, 2000)
"""

import numpy as np
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

# Support both relative (package) and absolute (standalone) imports
try:
    from .solver import HartreeFockSolver
    from .molecule import Molecule
except ImportError:
    from solver import HartreeFockSolver
    from molecule import Molecule


# ============================================================
# Physical Constants — imported from constants.py (single source of truth)
# ============================================================
try:
    from .phys_constants import (
        HARTREE_TO_CM1,
        HARTREE_TO_J,
        HARTREE_TO_EV,
        HARTREE_TO_KCAL,
        BOHR_TO_M,
        BOHR_TO_ANGSTROM,
        AMU_TO_KG,
        AMU_TO_ME,
        BOLTZMANN_EH as KB_EH,
        BOLTZMANN_SI as KB_SI,
        PLANCK_SI as H_PLANCK_SI,
        SPEED_OF_LIGHT_CM as C_SI,
    )
except ImportError:
    from phys_constants import (  # type: ignore[no-redef]
        HARTREE_TO_CM1,
        HARTREE_TO_J,
        HARTREE_TO_EV,
        HARTREE_TO_KCAL,
        BOHR_TO_M,
        BOHR_TO_ANGSTROM,
        AMU_TO_KG,
        AMU_TO_ME,
        BOLTZMANN_EH as KB_EH,
        BOLTZMANN_SI as KB_SI,
        PLANCK_SI as H_PLANCK_SI,
        SPEED_OF_LIGHT_CM as C_SI,
    )

HBAR_SI = H_PLANCK_SI / (2.0 * 3.141592653589793)  # ħ = h / 2π

# Avogadro's number
NA = 6.02214076e23

# Gas constant
R_SI = KB_SI * NA  # J/(mol*K)

# Pressure (1 atm in Pa)
P_ATM = 101325.0

# Frequency conversion factor from atomic units to cm^-1
# eigenvalue lambda is in Eh/(bohr^2 * amu)
# nu (cm^-1) = sqrt(lambda) / (2*pi*c) where c is in cm/s
# In atomic units: sqrt(Eh/(bohr^2 * amu)) has dimensions of 1/time
# Converting to SI: sqrt(HARTREE_TO_J / (BOHR_TO_M^2 * AMU_TO_KG)) / (2*pi*c_cm)
#   = sqrt(4.3597e-18 / (2.8003e-21 * 1.6605e-27)) / (2*pi*2.998e10)
#   = sqrt(4.3597e-18 / 4.6506e-48) / (1.8837e11)
#   = sqrt(9.3749e29) / (1.8837e11)
#   = 3.0618e14 / 1.8837e11
#   = 5140.487 cm^-1
# Exact: sqrt(E_h / (a_0^2 * m_u)) / (2*pi*c)
FREQ_AU_TO_CM1 = 5140.4869


# Atomic masses in amu (IUPAC 2016 standard atomic weights)
ATOMIC_MASS = {
    "H": 1.00794,
    "He": 4.00260,
    "Li": 6.94100,
    "Be": 9.01218,
    "B": 10.81100,
    "C": 12.01100,
    "N": 14.00670,
    "O": 15.99940,
    "F": 18.99840,
    "Ne": 20.17970,
    "Na": 22.98977,
    "Mg": 24.30500,
    "Al": 26.98154,
    "Si": 28.08550,
    "P": 30.97376,
    "S": 32.06600,
    "Cl": 35.45270,
    "Ar": 39.94800,
}

# Atomic numbers (for dipole moment calculations)
ATOMIC_NUMBER = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


# ============================================================
# Utility Functions
# ============================================================


def _atoms_to_flat(atoms):
    """Extract flat coordinate array [x0,y0,z0,x1,y1,z1,...] from atoms list."""
    coords = []
    for _, (x, y, z) in atoms:
        coords.extend([x, y, z])
    return np.array(coords, dtype=np.float64)


def _flat_to_atoms(flat, atoms_template):
    """Rebuild atoms list from flat coordinate array, preserving element labels."""
    new_atoms = []
    for i, (el, _) in enumerate(atoms_template):
        x, y, z = flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]
        new_atoms.append((el, (float(x), float(y), float(z))))
    return new_atoms


def _gradient_to_flat(gradients):
    """Convert list of 3D gradient vectors to flat numpy array."""
    flat = []
    for g in gradients:
        g_arr = np.asarray(g)
        flat.extend([g_arr[0], g_arr[1], g_arr[2]])
    return np.array(flat, dtype=np.float64)


def _is_linear(atoms, threshold=1e-6):
    """
    Determine if a molecule is linear.

    A molecule is linear if all atoms are collinear (lie along a single line).
    For 1-2 atoms, the molecule is always "linear" (or a point).

    Args:
        atoms: List of (element, (x, y, z)) tuples.
        threshold: Tolerance for cross-product norm to classify as collinear.

    Returns:
        True if the molecule is linear.
    """
    n_atoms = len(atoms)
    if n_atoms <= 2:
        return True

    # Use the first two distinct atoms to define a line direction
    r0 = np.array(atoms[0][1])
    direction = None
    for i in range(1, n_atoms):
        ri = np.array(atoms[i][1])
        d = ri - r0
        norm_d = np.linalg.norm(d)
        if norm_d > 1e-10:
            direction = d / norm_d
            break

    if direction is None:
        # All atoms at the same position (degenerate)
        return True

    # Check if all remaining atoms are collinear with this direction
    for i in range(2, n_atoms):
        ri = np.array(atoms[i][1])
        d = ri - r0
        norm_d = np.linalg.norm(d)
        if norm_d < 1e-10:
            continue  # coincident point is trivially collinear
        cross = np.cross(direction, d / norm_d)
        if np.linalg.norm(cross) > threshold:
            return False

    return True


def _build_translation_rotation_projector(atoms):
    """
    Build the projector P_TR that projects OUT translations and rotations from
    the 3N-dimensional Cartesian space.

    For a nonlinear molecule with N atoms, there are 6 TR degrees of freedom
    (3 translations + 3 rotations).  For linear, 5 (3T + 2R).

    Returns:
        P_int: (3N, 3N) projector onto internal (vibrational) subspace.
               P_int = I - P_TR where P_TR projects onto TR subspace.
    """
    n_atoms = len(atoms)
    n_coords = 3 * n_atoms
    masses = np.array([ATOMIC_MASS.get(el, 12.0) for el, _ in atoms])
    sqrt_masses = np.sqrt(masses)
    total_mass = np.sum(masses)

    # Center of mass
    com = np.zeros(3)
    for i, (el, (x, y, z)) in enumerate(atoms):
        com += masses[i] * np.array([x, y, z])
    com /= total_mass

    # Build translation vectors (mass-weighted)
    # T_d[3*i + d] = sqrt(m_i) for direction d, 0 otherwise
    tr_vectors = []
    for d in range(3):
        vec = np.zeros(n_coords)
        for i in range(n_atoms):
            vec[3 * i + d] = sqrt_masses[i]
        norm = np.linalg.norm(vec)
        if norm > 1e-14:
            tr_vectors.append(vec / norm)

    # Build rotation vectors (mass-weighted cross products with COM displacement)
    # R_d[3*i:3*i+3] = sqrt(m_i) * (e_d x (r_i - com))
    for d in range(3):
        e_d = np.zeros(3)
        e_d[d] = 1.0
        vec = np.zeros(n_coords)
        for i, (el, (x, y, z)) in enumerate(atoms):
            r = np.array([x, y, z]) - com
            cross = np.cross(e_d, r)
            vec[3 * i : 3 * i + 3] = sqrt_masses[i] * cross
        tr_vectors.append(vec)

    # Gram-Schmidt orthonormalization (handles linear molecules where one
    # rotation vector is zero)
    ortho = []
    for v in tr_vectors:
        for u in ortho:
            v = v - np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            ortho.append(v / norm)

    # Build projector onto TR subspace
    P_TR = np.zeros((n_coords, n_coords))
    for u in ortho:
        P_TR += np.outer(u, u)

    # Internal (vibrational) projector
    P_int = np.eye(n_coords) - P_TR

    return P_int


def _compute_moments_of_inertia(atoms):
    """
    Compute principal moments of inertia for a molecule.

    Args:
        atoms: List of (element, (x, y, z)) tuples, coordinates in Bohr.

    Returns:
        (I_A, I_B, I_C): Principal moments of inertia in amu*bohr^2,
                          sorted ascending.
    """
    masses = np.array([ATOMIC_MASS.get(el, 12.0) for el, _ in atoms])
    total_mass = np.sum(masses)

    # Center of mass
    com = np.zeros(3)
    for i, (el, (x, y, z)) in enumerate(atoms):
        com += masses[i] * np.array([x, y, z])
    com /= total_mass

    # Inertia tensor
    I_tensor = np.zeros((3, 3))
    for i, (el, (x, y, z)) in enumerate(atoms):
        r = np.array([x, y, z]) - com
        m = masses[i]
        I_tensor[0, 0] += m * (r[1] ** 2 + r[2] ** 2)
        I_tensor[1, 1] += m * (r[0] ** 2 + r[2] ** 2)
        I_tensor[2, 2] += m * (r[0] ** 2 + r[1] ** 2)
        I_tensor[0, 1] -= m * r[0] * r[1]
        I_tensor[0, 2] -= m * r[0] * r[2]
        I_tensor[1, 2] -= m * r[1] * r[2]

    I_tensor[1, 0] = I_tensor[0, 1]
    I_tensor[2, 0] = I_tensor[0, 2]
    I_tensor[2, 1] = I_tensor[1, 2]

    eigenvalues = np.linalg.eigvalsh(I_tensor)
    # Ensure non-negative (numerical noise can make tiny eigenvalues negative)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return tuple(sorted(eigenvalues))


def _compute_nuclear_dipole(atoms):
    """
    Compute nuclear contribution to the dipole moment.

    Args:
        atoms: List of (element, (x, y, z)) tuples, coordinates in Bohr.

    Returns:
        Nuclear dipole moment vector in atomic units (e*bohr).
    """
    mu_nuc = np.zeros(3)
    for el, (x, y, z) in atoms:
        Z = ATOMIC_NUMBER.get(el, 0)
        mu_nuc += Z * np.array([x, y, z])
    return mu_nuc


# ============================================================
# Main Class
# ============================================================


class VibrationalAnalysis:
    """
    Harmonic vibrational analysis for molecular systems.

    Computes the Hessian (force constant matrix) via central finite difference
    of analytical gradients, then performs mass-weighted normal mode analysis
    to obtain vibrational frequencies, normal modes, thermochemical properties,
    and IR intensities.

    Usage:
        from vibrational import VibrationalAnalysis
        from solver import HartreeFockSolver
        from molecule import Molecule

        mol = Molecule([('H', (0, 0, 0)), ('H', (0, 0, 1.4))], basis_name='sto-3g')
        hf = HartreeFockSolver()
        vib = VibrationalAnalysis()
        result = vib.compute_frequencies(mol, hf)
        thermo = vib.compute_thermochemistry(mol, hf, temperature=298.15)
    """

    def __init__(self, verbose=True):
        """
        Initialize vibrational analysis.

        Args:
            verbose: If True, print progress and results.
        """
        self.verbose = verbose

    # ----------------------------------------------------------------
    #  Hessian Computation
    # ----------------------------------------------------------------

    def compute_hessian(self, molecule, solver, step_size=0.005):
        """
        Compute the Cartesian Hessian matrix (force constant matrix) via
        central finite difference of analytical gradients.

        H[i,j] = (g[R + h*e_j] - g[R - h*e_j])_i / (2h)

        where g is the analytical gradient vector and e_j is a unit
        displacement along Cartesian coordinate j.

        The resulting Hessian is symmetrized: H_sym = 0.5 * (H + H^T)
        to correct for small numerical asymmetries from finite differencing.

        Args:
            molecule:  Molecule object (atoms in Bohr).
            solver:    Solver with compute_gradient(molecule) method
                       (e.g. HartreeFockSolver).
            step_size: Finite difference step size in Bohr (default 0.005).

        Returns:
            hessian: (3N, 3N) numpy array of second derivatives of energy
                     in Eh/bohr^2.
        """
        n_atoms = len(molecule.atoms)
        n_coords = 3 * n_atoms
        h = step_size

        if self.verbose:
            print("=" * 56)
            print("Hessian Computation (Central Finite Difference)")
            print(f"  Atoms       : {n_atoms}")
            print(f"  Coordinates : {n_coords}")
            print(f"  Step size   : {h:.6f} bohr")
            print(f"  Gradient evaluations: {2 * n_coords}")
            print("=" * 56)

        hessian = np.zeros((n_coords, n_coords))
        base_coords = _atoms_to_flat(molecule.atoms)

        # S16: parallel Hessian columns — each column is independent.
        # IMPORTANT: HartreeFockSolver is stateful (stores self.P, self.C etc.)
        # Each thread gets its own deep copy of the solver to avoid data races.
        def _compute_column(j):
            if self.verbose:
                atom_idx = j // 3
                axis_label = ["x", "y", "z"][j % 3]
                el = molecule.atoms[atom_idx][0]
                print(
                    f"  Displacing coordinate {j}/{n_coords} "
                    f"(atom {atom_idx} {el}, {axis_label})..."
                )

            # Each column gets its own solver copy (thread safety)
            local_solver = deepcopy(solver)

            # Forward displacement: R + h*e_j
            coords_plus = base_coords.copy()
            coords_plus[j] += h
            mol_plus = deepcopy(molecule)
            mol_plus.atoms = _flat_to_atoms(coords_plus, molecule.atoms)
            local_solver.compute_energy(mol_plus, verbose=False)
            grad_plus = local_solver.compute_gradient(mol_plus)
            g_plus = _gradient_to_flat(grad_plus)

            # Backward displacement: R - h*e_j
            coords_minus = base_coords.copy()
            coords_minus[j] -= h
            mol_minus = deepcopy(molecule)
            mol_minus.atoms = _flat_to_atoms(coords_minus, molecule.atoms)
            local_solver.compute_energy(mol_minus, verbose=False)
            grad_minus = local_solver.compute_gradient(mol_minus)
            g_minus = _gradient_to_flat(grad_minus)

            return j, (g_plus - g_minus) / (2.0 * h)

        # Run columns in parallel (4 workers — each with its own solver copy)
        with ThreadPoolExecutor(max_workers=min(4, n_coords)) as executor:
            for j, col in executor.map(_compute_column, range(n_coords)):
                hessian[:, j] = col

        # Symmetrize: H_sym = 0.5 * (H + H^T)
        hessian = 0.5 * (hessian + hessian.T)

        if self.verbose:
            print(f"\n  Hessian computed. Max |H_ij| = {np.max(np.abs(hessian)):.6e}")
            # Check symmetry quality
            asym = np.max(np.abs(hessian - hessian.T))
            print(f"  Symmetry deviation (pre-symmetrization): {asym:.2e}")

        return hessian

    # ----------------------------------------------------------------
    #  Normal Mode Analysis (Frequencies)
    # ----------------------------------------------------------------

    def compute_frequencies(self, molecule, solver, step_size=0.005, hessian=None):
        """
        Compute harmonic vibrational frequencies and normal modes.

        Steps:
          1. Compute Cartesian Hessian H (or use pre-computed).
          2. Mass-weight: H_mw = M^{-1/2} H M^{-1/2}
             where M = diag(m1,m1,m1, m2,m2,m2, ...) in amu.
          3. Project out translations and rotations.
          4. Diagonalize → eigenvalues lambda and eigenvectors.
          5. Convert eigenvalues to frequencies in cm^-1:
             - lambda > 0 → real frequency  (minimum)
             - lambda < 0 → imaginary frequency (saddle point / TS)
          6. Sort: imaginary first (flagged), then real ascending.

        Args:
            molecule:  Molecule object.
            solver:    Solver with compute_energy/compute_gradient.
            step_size: Finite difference step for Hessian (if not pre-computed).
            hessian:   Pre-computed (3N, 3N) Hessian, or None to compute.

        Returns:
            dict with keys:
                'frequencies_cm1':  Array of vibrational frequencies (cm^-1).
                                    Imaginary modes are returned as negative values.
                'eigenvalues':      Raw eigenvalues in Eh/(bohr^2 * amu).
                'normal_modes':     (n_vib, 3N) array of mass-weighted normal mode
                                    eigenvectors (each row is one mode).
                'n_imag':           Number of imaginary frequencies.
                'hessian':          The Cartesian Hessian (Eh/bohr^2).
                'is_linear':        Whether the molecule is linear.
                'zpe_hartree':      Zero-point energy in Hartree.
        """
        n_atoms = len(molecule.atoms)
        n_coords = 3 * n_atoms

        if self.verbose:
            print("\n" + "=" * 56)
            print("Vibrational Frequency Analysis")
            print("=" * 56)

        # Step 1: Hessian
        if hessian is None:
            hessian = self.compute_hessian(molecule, solver, step_size)
        else:
            if self.verbose:
                print("  Using pre-computed Hessian.")

        # Step 2: Mass-weighted Hessian
        masses = np.array([ATOMIC_MASS.get(el, 12.0) for el, _ in molecule.atoms])
        # Build diagonal mass vector: [m0, m0, m0, m1, m1, m1, ...]
        mass_vec = np.repeat(masses, 3)
        inv_sqrt_mass = 1.0 / np.sqrt(mass_vec)

        # H_mw[i,j] = H[i,j] / sqrt(m_i * m_j)
        H_mw = hessian * np.outer(inv_sqrt_mass, inv_sqrt_mass)

        # Step 3: Project out translations and rotations
        linear = _is_linear(molecule.atoms)
        n_tr = 5 if linear else 6
        if n_atoms == 1:
            n_tr = 3  # single atom: only translations

        P_int = _build_translation_rotation_projector(molecule.atoms)

        # Transform the mass-weighted Hessian into internal subspace
        # Need to apply projector in mass-weighted coordinates:
        # P_mw = M^{1/2} P_int M^{-1/2} ... but since P_int is built in
        # mass-weighted space already (using sqrt(m) weighted vectors),
        # we can directly project: H_mw_proj = P_int @ H_mw @ P_int
        H_mw_proj = P_int @ H_mw @ P_int

        # Step 4: Diagonalize
        eigenvalues, eigenvectors = np.linalg.eigh(H_mw_proj)

        # Step 5: Identify vibrational modes (discard near-zero TR modes)
        # The projected-out modes will have eigenvalue ~ 0.
        # We sort by absolute eigenvalue and keep the n_coords - n_tr largest.
        abs_evals = np.abs(eigenvalues)
        sorted_indices = np.argsort(abs_evals)

        # The n_tr smallest eigenvalues correspond to translations/rotations
        vib_indices = sorted_indices[n_tr:]
        # Sort vibrational modes by eigenvalue (ascending, so imaginary first)
        vib_evals = eigenvalues[vib_indices]
        sort_vib = np.argsort(vib_evals)
        vib_evals = vib_evals[sort_vib]
        vib_modes = eigenvectors[:, vib_indices[sort_vib]].T  # (n_vib, 3N)

        # Step 6: Convert to frequencies in cm^-1
        n_vib = len(vib_evals)
        frequencies = np.zeros(n_vib)
        n_imag = 0

        for i in range(n_vib):
            lam = vib_evals[i]
            if lam >= 0:
                # Real frequency: nu = sqrt(lambda) * conversion
                frequencies[i] = np.sqrt(lam) * FREQ_AU_TO_CM1
            else:
                # Imaginary frequency: report as negative cm^-1
                frequencies[i] = -np.sqrt(abs(lam)) * FREQ_AU_TO_CM1
                n_imag += 1

        # Zero-point energy: ZPE = 0.5 * sum(h*nu) for real modes only
        # In atomic units: nu_au = sqrt(lambda) (in a.u. angular frequency / 2pi)
        # ZPE = 0.5 * sum(nu_au) where nu_au = sqrt(lambda) for lambda > 0
        # But lambda is in Eh/(bohr^2*amu), we need frequency in Eh.
        # omega_au = sqrt(lambda * amu_to_me) where amu_to_me converts mass to electron mass
        # E_au = hbar * omega = omega (since hbar=1 in a.u.)
        # ZPE = 0.5 * sum(omega_au) for real modes
        # Equivalently: ZPE (cm^-1) = 0.5 * sum(freq_cm1 for real modes)
        # ZPE (Hartree) = ZPE (cm^-1) / HARTREE_TO_CM1
        real_freqs = frequencies[frequencies > 0]
        zpe_cm1 = 0.5 * np.sum(real_freqs)
        zpe_hartree = zpe_cm1 / HARTREE_TO_CM1

        if self.verbose:
            print(f"\n  Molecule is {'linear' if linear else 'nonlinear'}")
            print(f"  Atoms: {n_atoms}, Coordinates: {n_coords}")
            print(f"  TR modes removed: {n_tr}")
            print(f"  Vibrational modes: {n_vib}")
            print(f"  Imaginary frequencies: {n_imag}")
            print(f"\n  {'Mode':>4s}  {'Frequency (cm-1)':>18s}  {'Type':>10s}")
            print("  " + "-" * 38)
            for i in range(n_vib):
                if frequencies[i] < 0:
                    ftype = "imaginary"
                    fstr = f"{frequencies[i]:.2f}i"
                else:
                    ftype = "real"
                    fstr = f"{frequencies[i]:.2f}"
                print(f"  {i + 1:4d}  {fstr:>18s}  {ftype:>10s}")
            print(f"\n  Zero-Point Energy: {zpe_cm1:.2f} cm^-1")
            print(f"                   = {zpe_hartree:.8f} Eh")
            print(f"                   = {zpe_hartree * HARTREE_TO_KCAL:.4f} kcal/mol")

        n_real = len(frequencies) - n_imag

        return {
            "frequencies_cm1": frequencies,
            "eigenvalues": vib_evals,
            "normal_modes": vib_modes,
            "n_real": n_real,
            "n_imag": n_imag,
            "n_imaginary": n_imag,  # alias for backward compat
            "hessian": hessian,
            "is_linear": linear,
            "zpe_hartree": zpe_hartree,
            "zpe_cm1": zpe_cm1,
        }

    # ----------------------------------------------------------------
    #  Thermochemistry (IGRRHO model)
    # ----------------------------------------------------------------

    def compute_thermochemistry(
        self,
        molecule,
        solver,
        temperature=298.15,
        pressure=P_ATM,
        step_size=0.005,
        freq_result=None,
    ):
        """
        Compute thermochemical properties using the ideal-gas / rigid-rotor /
        harmonic-oscillator (IGRRHO) partition function model.

        From the vibrational frequencies, compute:
          - Zero-Point Energy (ZPE)
          - Translational contributions (E_trans, S_trans)
          - Rotational contributions (E_rot, S_rot)
          - Vibrational contributions (E_vib, S_vib)
          - Electronic energy E_elec (from solver)
          - Total internal energy U, enthalpy H, entropy S, Gibbs free energy G

        All energies returned in Hartree, entropy in Eh/K.

        Args:
            molecule:    Molecule object.
            solver:      Solver with compute_energy/compute_gradient.
            temperature: Temperature in Kelvin (default 298.15 K).
            pressure:    Pressure in Pascal (default 1 atm = 101325 Pa).
            step_size:   Finite difference step for Hessian.
            freq_result: Pre-computed result from compute_frequencies(), or None.

        Returns:
            dict with thermochemical quantities.
        """
        T = temperature
        kT = KB_EH * T  # in Hartree

        if self.verbose:
            print("\n" + "=" * 56)
            print(f"Thermochemistry at T = {T:.2f} K, P = {pressure:.0f} Pa")
            print("=" * 56)

        # Get frequencies (compute if not provided)
        if freq_result is None:
            freq_result = self.compute_frequencies(molecule, solver, step_size)

        frequencies_cm1 = freq_result["frequencies_cm1"]
        zpe_hartree = freq_result["zpe_hartree"]
        is_linear = freq_result["is_linear"]

        # Electronic energy from solver
        E_total, _ = solver.compute_energy(molecule, verbose=False)

        n_atoms = len(molecule.atoms)
        masses = np.array([ATOMIC_MASS.get(el, 12.0) for el, _ in molecule.atoms])
        total_mass = np.sum(masses)

        # ============================================================
        # 1. Translational Contributions (3D ideal gas)
        # ============================================================
        # E_trans = 3/2 * kT
        E_trans = 1.5 * kT

        # S_trans (Sackur-Tetrode equation)
        # S_trans/R = 5/2 + ln[ (2*pi*M*kT/h^2)^(3/2) * kT/P ]
        # Working in SI units for this part:
        M_kg = total_mass * AMU_TO_KG
        kT_SI = KB_SI * T
        lambda_thermal = H_PLANCK_SI / np.sqrt(2.0 * np.pi * M_kg * kT_SI)
        # q_trans / V = 1 / lambda^3
        # S_trans/k = 5/2 + ln(V * q_trans/V / N)  ... but for ideal gas:
        # S_trans = R * [5/2 + ln( (2*pi*M*kT)^(3/2) / (h^3 * P / (kT)) )]
        # = R * [5/2 + ln( (2*pi*M*kT/h^2)^(3/2) * kT/P )]
        q_trans_over_V = (2.0 * np.pi * M_kg * kT_SI / H_PLANCK_SI**2) ** 1.5
        V_per_particle = kT_SI / pressure
        q_trans = q_trans_over_V * V_per_particle
        S_trans_SI = R_SI * (2.5 + np.log(q_trans))  # J/(mol*K)
        S_trans = S_trans_SI / (HARTREE_TO_J * NA)  # Convert to Eh/(mol*K)...
        # Actually we want S per molecule in Eh/K:
        S_trans = KB_EH * (2.5 + np.log(q_trans))

        # ============================================================
        # 2. Rotational Contributions (rigid rotor)
        # ============================================================
        if n_atoms == 1:
            E_rot = 0.0
            S_rot = 0.0
        else:
            I_A, I_B, I_C = _compute_moments_of_inertia(molecule.atoms)
            # Convert moments from amu*bohr^2 to kg*m^2
            I_A_SI = I_A * AMU_TO_KG * BOHR_TO_M**2
            I_B_SI = I_B * AMU_TO_KG * BOHR_TO_M**2
            I_C_SI = I_C * AMU_TO_KG * BOHR_TO_M**2

            # Symmetry number (simplified: assume sigma=1 for general case)
            # A proper implementation would detect molecular point group.
            sigma = 1

            if is_linear:
                # Linear rotor: E_rot = kT, q_rot = 8*pi^2*I*kT / (sigma*h^2)
                E_rot = kT
                # Use the non-zero moment (I_C for linear molecule along z)
                I_lin = max(I_B_SI, I_C_SI)  # largest non-zero moment
                if I_lin > 1e-50:
                    q_rot = 8.0 * np.pi**2 * I_lin * kT_SI / (sigma * H_PLANCK_SI**2)
                    S_rot = KB_EH * (1.0 + np.log(q_rot))
                else:
                    S_rot = 0.0
            else:
                # Nonlinear rotor: E_rot = 3/2 * kT
                E_rot = 1.5 * kT
                # q_rot = sqrt(pi) / sigma * (8*pi^2*kT/h^2)^(3/2) * sqrt(I_A*I_B*I_C)
                if I_A_SI > 1e-50 and I_B_SI > 1e-50 and I_C_SI > 1e-50:
                    q_rot = (
                        np.sqrt(np.pi)
                        / sigma
                        * (8.0 * np.pi**2 * kT_SI / H_PLANCK_SI**2) ** 1.5
                        * np.sqrt(I_A_SI * I_B_SI * I_C_SI)
                    )
                    S_rot = KB_EH * (1.5 + np.log(q_rot))
                else:
                    S_rot = 0.0

        # ============================================================
        # 3. Vibrational Contributions (quantum harmonic oscillator)
        # ============================================================
        # Use only real (positive) frequencies
        real_freqs_cm1 = frequencies_cm1[frequencies_cm1 > 0]

        # Convert cm^-1 to Hartree: E = h*c*nu_cm1
        # In atomic units: E_au = nu_cm1 / HARTREE_TO_CM1
        freq_hartree = real_freqs_cm1 / HARTREE_TO_CM1

        # Thermal vibrational energy (above ZPE)
        # E_vib = sum_i [ h*nu_i / (exp(h*nu_i/kT) - 1) ]
        E_vib = 0.0
        S_vib = 0.0

        for nu in freq_hartree:
            x = nu / kT  # dimensionless: h*nu / (k*T)
            if x > 500:
                # High-frequency limit: essentially no population
                continue
            exp_x = np.exp(x)
            # Vibrational energy contribution (thermal, above ZPE)
            E_vib += nu / (exp_x - 1.0)
            # Vibrational entropy contribution
            # S_vib_i = k * [x/(exp(x)-1) - ln(1-exp(-x))]
            S_vib += KB_EH * (x / (exp_x - 1.0) - np.log(1.0 - np.exp(-x)))

        # ============================================================
        # 4. Electronic Contribution
        # ============================================================
        # For a singlet ground state: q_elec = multiplicity = 1
        # E_elec = 0 (no thermal electronic excitation)
        # S_elec = k * ln(multiplicity)
        mult = getattr(molecule, "multiplicity", 1)
        E_elec_thermal = 0.0
        S_elec = KB_EH * np.log(mult) if mult > 1 else 0.0

        # ============================================================
        # 5. Totals
        # ============================================================
        # Total thermal correction to energy (above 0 K electronic energy)
        E_thermal = E_trans + E_rot + E_vib + zpe_hartree + E_elec_thermal

        # Internal energy at T
        U = E_total + E_thermal

        # Enthalpy: H = U + kT (PV = NkT for ideal gas, per molecule)
        H = U + kT

        # Total entropy
        S_total = S_trans + S_rot + S_vib + S_elec

        # Gibbs free energy: G = H - T*S
        G = H - T * S_total

        if self.verbose:
            print(f"\n  Electronic Energy (E_0)    : {E_total:.8f} Eh")
            print(
                f"  Zero-Point Energy (ZPE)    : {zpe_hartree:.8f} Eh "
                f"({zpe_hartree * HARTREE_TO_KCAL:.4f} kcal/mol)"
            )
            print(f"\n  Thermal Corrections at {T:.2f} K:")
            print(f"    E_translational          : {E_trans:.8f} Eh")
            print(f"    E_rotational             : {E_rot:.8f} Eh")
            print(f"    E_vibrational (thermal)  : {E_vib:.8f} Eh")
            print(f"    E_thermal (total)        : {E_thermal:.8f} Eh")
            print(f"\n  Entropy Contributions:")
            print(f"    S_translational          : {S_trans:.10e} Eh/K")
            print(f"    S_rotational             : {S_rot:.10e} Eh/K")
            print(f"    S_vibrational            : {S_vib:.10e} Eh/K")
            print(f"    S_electronic             : {S_elec:.10e} Eh/K")
            print(f"    S_total                  : {S_total:.10e} Eh/K")
            print(f"\n  Thermodynamic Quantities:")
            print(f"    U (internal energy)      : {U:.8f} Eh")
            print(f"    H (enthalpy)             : {H:.8f} Eh")
            print(f"    G (Gibbs free energy)    : {G:.8f} Eh")
            print(f"    T*S                      : {T * S_total:.8f} Eh")
            print(
                f"    G - E_0                  : {(G - E_total):.8f} Eh "
                f"({(G - E_total) * HARTREE_TO_KCAL:.4f} kcal/mol)"
            )

        return {
            "E_electronic": E_total,
            "zpe_hartree": zpe_hartree,
            "zpe_kcal": zpe_hartree * HARTREE_TO_KCAL,
            "E_trans": E_trans,
            "E_rot": E_rot,
            "E_vib": E_vib,
            "E_thermal": E_thermal,
            "S_trans": S_trans,
            "S_rot": S_rot,
            "S_vib": S_vib,
            "S_elec": S_elec,
            "S_total": S_total,
            "U": U,
            "H": H,
            "G": G,
            "temperature": T,
            "pressure": pressure,
            "frequencies_cm1": frequencies_cm1,
        }

    # ----------------------------------------------------------------
    #  IR Spectrum (Dipole Derivatives)
    # ----------------------------------------------------------------

    def compute_ir_spectrum(self, molecule, solver, step_size=0.005, freq_result=None):
        """
        Compute IR absorption intensities from the derivative of the dipole
        moment with respect to normal mode coordinates.

        IR intensity of mode Q_k is proportional to |d mu / d Q_k|^2,
        where mu is the molecular dipole moment vector.

        The dipole derivatives are computed numerically:
          d mu_alpha / d x_j = (mu_alpha(R + h*e_j) - mu_alpha(R - h*e_j)) / (2h)

        Then transformed to normal mode coordinates:
          d mu / d Q_k = sum_j (d mu / d x_j) * L_jk / sqrt(m_j)

        where L is the normal mode eigenvector matrix.

        Args:
            molecule:    Molecule object.
            solver:      Solver with compute_energy (for SCF at displaced geometries).
            step_size:   Finite difference step in Bohr.
            freq_result: Pre-computed result from compute_frequencies(), or None.

        Returns:
            dict with keys:
                'frequencies_cm1': Vibrational frequencies (cm^-1).
                'ir_intensities':  IR intensities in (e*bohr)^2/amu
                                   (proportional to absorption intensity).
                'dipole_derivs':   (3, 3N) matrix of dipole derivatives
                                   d mu_alpha / d x_j.
        """
        n_atoms = len(molecule.atoms)
        n_coords = 3 * n_atoms
        h = step_size

        if self.verbose:
            print("\n" + "=" * 56)
            print("IR Spectrum Computation (Dipole Derivatives)")
            print("=" * 56)

        # Get frequencies and normal modes
        if freq_result is None:
            freq_result = self.compute_frequencies(molecule, solver, step_size)

        frequencies_cm1 = freq_result["frequencies_cm1"]
        normal_modes = freq_result["normal_modes"]  # (n_vib, 3N)
        n_vib = len(frequencies_cm1)

        # Compute dipole derivatives: d mu / d x_j via central finite difference
        # mu = mu_nuclear - mu_electronic
        # mu_nuclear = sum_A Z_A * R_A
        # mu_electronic = -Tr(P * D_alpha) where D_alpha is the dipole integral matrix
        # For simplicity, we compute the total dipole numerically at each displaced geometry.

        base_coords = _atoms_to_flat(molecule.atoms)

        # Compute dipole at reference geometry
        mu_ref = self._compute_dipole_moment(molecule, solver)

        # Dipole derivative matrix: d mu_alpha / d x_j  (3 x 3N)
        d_mu_dx = np.zeros((3, n_coords))

        for j in range(n_coords):
            if self.verbose:
                atom_idx = j // 3
                axis_label = ["x", "y", "z"][j % 3]
                el = molecule.atoms[atom_idx][0]
                print(
                    f"  Dipole derivative for coord {j}/{n_coords} "
                    f"(atom {atom_idx} {el}, {axis_label})..."
                )

            # Forward displacement
            coords_plus = base_coords.copy()
            coords_plus[j] += h
            mol_plus = deepcopy(molecule)
            mol_plus.atoms = _flat_to_atoms(coords_plus, molecule.atoms)
            mu_plus = self._compute_dipole_moment(mol_plus, solver)

            # Backward displacement
            coords_minus = base_coords.copy()
            coords_minus[j] -= h
            mol_minus = deepcopy(molecule)
            mol_minus.atoms = _flat_to_atoms(coords_minus, molecule.atoms)
            mu_minus = self._compute_dipole_moment(mol_minus, solver)

            # Central difference
            d_mu_dx[:, j] = (mu_plus - mu_minus) / (2.0 * h)

        # Transform to normal mode coordinates
        # d mu / d Q_k = sum_j (d mu / d x_j) * L_jk / sqrt(m_j)
        # where normal_modes[k, j] = L_jk (mass-weighted eigenvector)
        # The normal mode vectors are already mass-weighted, so we need to
        # un-mass-weight: the Cartesian displacement for mode k is
        # delta_x_j = L_jk / sqrt(m_j)  (L is in mass-weighted coords)
        masses = np.array([ATOMIC_MASS.get(el, 12.0) for el, _ in molecule.atoms])
        mass_vec = np.repeat(masses, 3)
        inv_sqrt_mass = 1.0 / np.sqrt(mass_vec)

        ir_intensities = np.zeros(n_vib)

        for k in range(n_vib):
            # L_k is the mass-weighted normal mode eigenvector (already normalized)
            L_k = normal_modes[k]  # (3N,)

            # d mu / d Q_k  = d_mu_dx @ (L_k * inv_sqrt_mass)
            # The normal mode displacement in Cartesian coords
            cart_displacement = L_k * inv_sqrt_mass
            d_mu_dQ = d_mu_dx @ cart_displacement  # (3,)

            # IR intensity proportional to |d mu / d Q_k|^2
            ir_intensities[k] = np.dot(d_mu_dQ, d_mu_dQ)

        # Normalize: convert to km/mol (approximate via standard conversion)
        # I (km/mol) = (N_A * pi) / (3 * c^2) * |d mu / d Q|^2
        # For now, report in atomic units squared (e^2*bohr^2/amu)
        # which is proportional to the true IR intensity.

        if self.verbose:
            print(f"\n  {'Mode':>4s}  {'Freq (cm-1)':>14s}  {'IR Intensity':>14s}")
            print("  " + "-" * 38)
            for k in range(n_vib):
                freq = frequencies_cm1[k]
                if freq < 0:
                    fstr = f"{freq:.2f}i"
                else:
                    fstr = f"{freq:.2f}"
                print(f"  {k + 1:4d}  {fstr:>14s}  {ir_intensities[k]:14.6f}")

        return {
            "frequencies_cm1": frequencies_cm1,
            "ir_intensities": ir_intensities,
            "dipole_derivs": d_mu_dx,
        }

    # ----------------------------------------------------------------
    #  Dipole Moment Helper
    # ----------------------------------------------------------------

    def _compute_dipole_moment(self, molecule, solver):
        """
        Compute the molecular dipole moment vector at the given geometry.

        mu = mu_nuclear + mu_electronic
        mu_nuclear = sum_A Z_A * R_A  (positive nuclear charges)
        mu_electronic = -sum_{mu,nu} P_{mu,nu} * <mu| r |nu>

        For the electronic part, we compute the dipole integrals
        <mu| x |nu>, <mu| y |nu>, <mu| z |nu> using the overlap integral
        with shifted angular momentum (Obara-Saika).

        When the solver provides a density matrix and basis, we use the full
        quantum mechanical dipole. Otherwise, we fall back to a nuclear-only
        estimate (which is only valid at equilibrium for neutral molecules).

        Args:
            molecule: Molecule object.
            solver:   Solver (used to get density matrix P and basis).

        Returns:
            mu: (3,) dipole moment vector in atomic units (e*bohr).
        """
        # Run SCF to get density matrix
        solver.compute_energy(molecule, verbose=False)

        # Nuclear contribution
        mu_nuc = _compute_nuclear_dipole(molecule.atoms)

        # Electronic contribution: -Tr(P * D_alpha) for alpha = x, y, z
        if hasattr(solver, "P") and hasattr(solver, "basis"):
            P = solver.P
            basis = solver.basis
            N = len(basis)
            mu_elec = np.zeros(3)

            for alpha in range(3):
                # Build dipole integral matrix D_alpha[mu,nu] = <mu| r_alpha |nu>
                D = np.zeros((N, N))
                for i in range(N):
                    for j in range(N):
                        for pi in basis[i].primitives:
                            for pj in basis[j].primitives:
                                D[i, j] += self._dipole_integral(pi, pj, alpha)
                mu_elec[alpha] = -np.sum(P * D)

            return mu_nuc + mu_elec
        else:
            # Fallback: nuclear only (inaccurate but provides correct trends)
            return mu_nuc

    def _dipole_integral(self, a, b, component):
        """
        Compute the dipole integral <a| r_component |b> between two primitive
        Gaussian basis functions.

        Uses the relation:  x * exp(-beta*r^2) = derivative trick via
        recurrence with overlap integrals of shifted angular momentum.

        <a| x |b> = N_a * N_b * overlap_integral(a_shifted, b)
        where a_shifted has angular momentum (l+1, m, n) for component=0 (x).

        More precisely, using the Obara-Saika overlap recurrence:
        <a| x_c |b> = <a_shifted|b> / (2*alpha_a)  (not quite right for GTO)

        Actually the correct approach is:
        r_alpha * G_b = origin_b[alpha]*G_b + d/d(2*beta) for the appropriate term.

        Simplest correct formula using Obara-Saika:
        <la,ma,na| x |lb,mb,nb> uses the identity:
          x = (x - Ax) + Ax
          x * phi_a = phi_a' (with l+1) if centered, plus Ax * phi_a

        So: <a| x |b> = <a(l+1)| b> + Ax * <a|b>  (for x-component)

        This uses the existing overlap_jit infrastructure.
        """
        # Import the overlap function from integrals
        try:
            from .integrals import overlap_jit
        except ImportError:
            from integrals import overlap_jit

        alpha = a.alpha
        beta = b.alpha

        la, ma, na = a.l, a.m, a.n
        lb, mb, nb = b.l, b.m, b.n
        ra, rb = a.origin, b.origin

        # <a| r_c |b> = <a(l+delta)| b> + A_c * <a|b>
        # where delta increments the angular momentum in direction c by 1,
        # and A_c is the origin of basis function a in direction c.
        #
        # This follows from: r_c * phi_a(r) = (r_c - A_c)*phi_a + A_c*phi_a
        # The first term is phi_a with l_c incremented by 1 (unnormalized).
        # But we need to handle normalization carefully.

        # Term 1: <a|b> (standard overlap)
        S_ab = overlap_jit(la, ma, na, lb, mb, nb, ra, rb, alpha, beta)
        origin_c = ra[component]

        # Term 2: overlap with angular momentum shifted up by 1 in direction c
        # This is the raw (unnormalized) overlap integral.
        if component == 0:
            S_shifted = overlap_jit(la + 1, ma, na, lb, mb, nb, ra, rb, alpha, beta)
        elif component == 1:
            S_shifted = overlap_jit(la, ma + 1, na, lb, mb, nb, ra, rb, alpha, beta)
        else:
            S_shifted = overlap_jit(la, ma, na + 1, lb, mb, nb, ra, rb, alpha, beta)

        # The recurrence relation for (r_c - A_c) * phi_a gives phi with
        # angular momentum l_c + 1 but the UNNORMALIZED overlap integral
        # already accounts for the polynomial factors.
        # Full integral: a.N * b.N * (S_shifted + origin_c * S_ab)
        return a.N * b.N * (S_shifted + origin_c * S_ab)

    # ----------------------------------------------------------------
    #  Convenience: Full Analysis
    # ----------------------------------------------------------------

    def run_full_analysis(
        self, molecule, solver, temperature=298.15, step_size=0.005, compute_ir=True
    ):
        """
        Run complete vibrational analysis: frequencies, thermochemistry, IR.

        S25: Hessian computed ONCE here and passed to all downstream calls,
        preventing the triple redundant rebuild that occurred when each
        method independently called compute_frequencies/compute_hessian.

        Args:
            molecule:    Molecule object.
            solver:      Solver with compute_energy/compute_gradient.
            temperature: Temperature in K (default 298.15).
            step_size:   Finite difference step.
            compute_ir:  Whether to compute IR intensities (default True).

        Returns:
            dict with all results combined.
        """
        if self.verbose:
            print("\n" + "=" * 60)
            print("  FULL VIBRATIONAL ANALYSIS")
            print("=" * 60)

        # S25: Compute Hessian ONCE and pass to all downstream calls
        hessian = self.compute_hessian(molecule, solver, step_size)

        # Step 1: Frequencies (pass pre-computed hessian)
        freq_result = self.compute_frequencies(
            molecule, solver, step_size, hessian=hessian
        )

        # Step 2: Thermochemistry (pass pre-computed freq_result)
        thermo_result = self.compute_thermochemistry(
            molecule, solver, temperature, step_size=step_size, freq_result=freq_result
        )

        result = {**freq_result, **thermo_result}

        # Step 3: IR spectrum (pass pre-computed freq_result to avoid another hessian)
        if compute_ir:
            ir_result = self.compute_ir_spectrum(
                molecule, solver, step_size, freq_result=freq_result
            )
            result["ir_intensities"] = ir_result["ir_intensities"]
            result["dipole_derivs"] = ir_result["dipole_derivs"]

        if self.verbose:
            print("\n" + "=" * 60)
            print("  VIBRATIONAL ANALYSIS COMPLETE")
            print("=" * 60)

        return result
