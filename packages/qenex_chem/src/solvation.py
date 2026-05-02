"""
Polarizable Continuum Model (PCM) — Implicit Solvation
=======================================================

Implements the Conductor-like PCM (CPCM) for modeling solvation effects.

The solvent is represented as a dielectric continuum surrounding a molecular
cavity constructed from interlocking van der Waals spheres. The solute-solvent
electrostatic interaction is captured by apparent surface charges (ASC) placed
on the cavity boundary.

Theory:
    1. Build molecular cavity from scaled vdW atomic spheres
    2. Discretize cavity surface into tesserae (surface elements)
    3. Compute electrostatic potential V at each tessera from solute charges
    4. Solve CPCM linear equation:  q = -f(eps) * D^{-1} * V
       where f(eps) = (eps - 1) / eps  (conductor-like screening)
       and D is the Coulomb interaction matrix between tesserae
    5. Solvation energy:  E_solv = 0.5 * sum(q_i * V_i)

The CPCM approximation treats the solvent as a conductor (eps -> infinity)
then scales by f(eps). This is exact for high dielectric solvents (water)
and a good approximation for moderate dielectrics.

Surface discretization:
    - Atom-centered spheres with Lebedev angular quadrature
    - Solvent-excluded surface: remove tesserae inside neighboring spheres
    - Self-interaction diagonal: D[i,i] = 1.07 * sqrt(4*pi / a_i)

References:
    Tomasi, J. et al. Chem. Rev. 105, 2999 (2005) — PCM review
    Barone, V. & Cossi, M. J. Phys. Chem. A 102, 1995 (1998) — CPCM
    Cossi, M. et al. J. Comput. Chem. 24, 669 (2003) — cavity construction
    Klamt, A. & Schuurmann, G. J. Chem. Soc. Perkin Trans. 2, 799 (1993) — COSMO

Validated:
    - Cavity construction: tesserae lie on atomic sphere surfaces
    - Normal vectors: outward-pointing, unit length
    - Coulomb matrix: symmetric, positive-definite
    - Surface charges: sum to -f(eps) * Q_solute (Gauss' law)
    - Solvation energy: negative for polar solutes in polar solvents
    - Vacuum (eps=1.0): E_solv = 0 identically

Usage:
    from solvation import PCMSolver
    from solver import HartreeFockSolver
    from molecule import Molecule

    mol = Molecule([('O', (0,0,0)), ('H', (0,0,1.8))], basis_name='sto-3g')
    hf = HartreeFockSolver()
    hf.compute_energy(mol)

    pcm = PCMSolver(solvent='water')
    E_solv = pcm.compute_solvation_energy(mol, hf)
    E_total = pcm.compute_solvated_energy(mol, hf)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Support both package and direct imports
try:
    from .molecule import Molecule
except ImportError:
    from molecule import Molecule


# ===========================================================================
# Physical Constants
# ===========================================================================

# Physical constants — from constants.py (single source of truth)
try:
    from .phys_constants import (
        BOHR_TO_ANGSTROM,
        ANGSTROM_TO_BOHR,
        HARTREE_TO_EV,
        HARTREE_TO_KCAL,
    )
except ImportError:
    from phys_constants import (
        BOHR_TO_ANGSTROM,
        ANGSTROM_TO_BOHR,
        HARTREE_TO_EV,
        HARTREE_TO_KCAL,
    )  # type: ignore[no-redef]


# ===========================================================================
# Solvent Database
# ===========================================================================

SOLVENTS: Dict[str, Dict[str, Any]] = {
    "water": {"epsilon": 78.39, "name": "Water"},
    "methanol": {"epsilon": 32.70, "name": "Methanol"},
    "ethanol": {"epsilon": 24.55, "name": "Ethanol"},
    "dmso": {"epsilon": 46.70, "name": "DMSO"},
    "acetonitrile": {"epsilon": 37.50, "name": "Acetonitrile"},
    "chloroform": {"epsilon": 4.81, "name": "Chloroform"},
    "benzene": {"epsilon": 2.27, "name": "Benzene"},
    "toluene": {"epsilon": 2.38, "name": "Toluene"},
    "thf": {"epsilon": 7.58, "name": "THF"},
    "dcm": {"epsilon": 8.93, "name": "Dichloromethane"},
    "vacuum": {"epsilon": 1.00, "name": "Vacuum (gas phase)"},
}


# ===========================================================================
# Van der Waals Radii (Angstrom)
# ===========================================================================
# Standard Bondi/UFF radii used in PCM cavity construction.
# Ref: Bondi, A. J. Phys. Chem. 68, 441 (1964)

VDW_RADII_ANGSTROM: Dict[str, float] = {
    "H": 1.20,
    "He": 1.40,
    "Li": 1.82,
    "Be": 1.53,
    "B": 1.92,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Al": 1.84,
    "Si": 2.10,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Ar": 1.88,
}

# Atomic numbers for nuclear charge lookup
_Z_MAP: Dict[str, int] = {
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

# Default cavity scaling factor (standard in PCM literature)
DEFAULT_CAVITY_SCALING = 1.2


# ===========================================================================
# Lebedev Angular Quadrature for Cavity Surface Tessellation
# ===========================================================================


def _lebedev_sphere(n_points: int = 110) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate angular grid points and weights on the unit sphere.

    Uses Lebedev quadrature for accurate integration of spherical harmonics.
    Falls back to a uniform angular grid (theta, phi) for arbitrary point counts.

    For the 26-point Lebedev grid, uses analytically known points and weights
    (exact for l <= 7). For other sizes, uses a Fibonacci spiral approximation
    that gives good uniformity for cavity surface discretization.

    Args:
        n_points: Desired number of angular grid points.
                  26 = Lebedev analytical (degree 7)
                  Other = Fibonacci spiral (approximately uniform)

    Returns:
        Tuple of (points, weights) where:
            points: (n_points, 3) array of unit vectors
            weights: (n_points,) array, sum = 4*pi
    """
    if n_points == 26:
        return _lebedev_26()
    else:
        return _fibonacci_sphere(n_points)


def _lebedev_26() -> Tuple[np.ndarray, np.ndarray]:
    """
    Lebedev 26-point angular quadrature (exact for l <= 7).

    Three orbit types on the unit sphere:
        - 6 octahedral vertices  (weight A1)
        - 12 edge midpoints      (weight A2)
        - 8 cube vertices        (weight A3)

    Ref: Lebedev, Zh. Vychisl. Mat. Mat. Fiz. 16 (1976) 293-306
    """
    a2 = 1.0 / np.sqrt(2.0)
    a3 = 1.0 / np.sqrt(3.0)

    points = np.array(
        [
            # 6 octahedral vertices
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
            # 12 edge midpoints
            [a2, a2, 0],
            [-a2, a2, 0],
            [a2, -a2, 0],
            [-a2, -a2, 0],
            [a2, 0, a2],
            [-a2, 0, a2],
            [a2, 0, -a2],
            [-a2, 0, -a2],
            [0, a2, a2],
            [0, -a2, a2],
            [0, a2, -a2],
            [0, -a2, -a2],
            # 8 cube vertices
            [a3, a3, a3],
            [-a3, a3, a3],
            [a3, -a3, a3],
            [a3, a3, -a3],
            [-a3, -a3, a3],
            [-a3, a3, -a3],
            [a3, -a3, -a3],
            [-a3, -a3, -a3],
        ],
        dtype=np.float64,
    )

    # Weights: sum = 4*pi
    w1 = 4.0 * np.pi / 21.0  # octahedral vertices
    w2 = 4.0 * np.pi * 4.0 / 105.0  # edge midpoints
    w3 = 4.0 * np.pi * 27.0 / 840.0  # cube vertices

    weights = np.concatenate(
        [
            np.full(6, w1),
            np.full(12, w2),
            np.full(8, w3),
        ]
    )

    return points, weights


def _fibonacci_sphere(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate approximately uniform points on the unit sphere using a
    Fibonacci spiral (golden spiral) construction.

    This gives excellent uniformity for cavity surface discretization
    where exact spherical harmonic integration is not required.

    Each point is assigned equal area: 4*pi / n_points.

    Args:
        n_points: Number of points to generate (must be >= 6)

    Returns:
        Tuple of (points, weights) on the unit sphere
    """
    n_points = max(n_points, 6)

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(n_points, dtype=np.float64)

    # Latitude: uniformly spaced in cos(theta)
    # theta_i = arccos(1 - 2*(i + 0.5)/N)
    cos_theta = 1.0 - 2.0 * (indices + 0.5) / n_points
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Longitude: golden angle increments
    phi = 2.0 * np.pi * indices / golden_ratio

    sin_theta = np.sin(theta)
    points = np.column_stack(
        [
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        ]
    )

    # Equal-area weights
    weights = np.full(n_points, 4.0 * np.pi / n_points)

    return points, weights


# ===========================================================================
# Cavity Surface Data Structure
# ===========================================================================


class CavitySurface:
    """
    Discretized molecular cavity surface.

    Stores the tessellation of the solvent-excluded surface (SES) around
    a molecule, consisting of surface elements (tesserae) with positions,
    outward normal vectors, and area elements.

    Attributes:
        positions:  (n_tess, 3) array — tessera center coordinates (Bohr)
        normals:    (n_tess, 3) array — outward unit normal vectors
        areas:      (n_tess,) array   — tessera area elements (Bohr^2)
        atom_idx:   (n_tess,) array   — index of the parent atom for each tessera
        n_tesserae: int               — total number of surface elements
    """

    def __init__(
        self,
        positions: np.ndarray,
        normals: np.ndarray,
        areas: np.ndarray,
        atom_idx: np.ndarray,
    ):
        self.positions = positions
        self.normals = normals
        self.areas = areas
        self.atom_idx = atom_idx
        self.n_tesserae = len(areas)

    def __repr__(self) -> str:
        total_area = np.sum(self.areas)
        return (
            f"CavitySurface(n_tesserae={self.n_tesserae}, "
            f"total_area={total_area:.2f} bohr^2)"
        )


# ===========================================================================
# PCM Solver
# ===========================================================================


class PCMSolver:
    """
    Conductor-like Polarizable Continuum Model (CPCM) solver.

    Models solvation by embedding the solute in a dielectric continuum.
    The solute-solvent interaction is captured by apparent surface charges
    on a molecular cavity constructed from van der Waals spheres.

    The CPCM equation:
        q = -f(eps) * D^{-1} * V

    where:
        f(eps) = (eps - 1) / eps    — conductor-like screening factor
        D[i,j] = 1 / |r_i - r_j|   — Coulomb interaction between tesserae
        D[i,i] = 1.07 * sqrt(4*pi / a_i)  — self-interaction (Cossi et al.)
        V[i]   = electrostatic potential at tessera i from solute charges
        q[i]   = apparent surface charge on tessera i

    Solvation energy:
        E_solv = 0.5 * sum(q_i * V_i)

    Usage:
        pcm = PCMSolver(solvent='water', cavity_type='vdw')
        cavity = pcm.build_cavity(molecule)
        E_solv = pcm.compute_solvation_energy(molecule, hf_solver)
        E_total = pcm.compute_solvated_energy(molecule, hf_solver)
    """

    def __init__(
        self,
        solvent: str = "water",
        cavity_type: str = "vdw",
        scaling: float = DEFAULT_CAVITY_SCALING,
        n_angular: int = 110,
    ):
        """
        Initialize PCM solver.

        Args:
            solvent:     Solvent name (key in SOLVENTS dict) or numeric epsilon
            cavity_type: Cavity construction method ('vdw' = van der Waals)
            scaling:     Scaling factor for vdW radii (default 1.2, standard in PCM)
            n_angular:   Number of angular grid points per atomic sphere
                         (26, 110, or any positive integer via Fibonacci spiral)
        """
        # Parse solvent
        if isinstance(solvent, (int, float)):
            self.epsilon = float(solvent)
            self.solvent_name = f"Custom (eps={self.epsilon:.2f})"
        elif solvent.lower() in SOLVENTS:
            self.epsilon = SOLVENTS[solvent.lower()]["epsilon"]
            self.solvent_name = SOLVENTS[solvent.lower()]["name"]
        else:
            raise ValueError(
                f"Unknown solvent '{solvent}'. Available: {list(SOLVENTS.keys())} "
                f"or pass a numeric dielectric constant."
            )

        if self.epsilon < 1.0:
            raise ValueError(
                f"Dielectric constant must be >= 1.0 (got {self.epsilon}). "
                f"Use 'vacuum' for gas phase."
            )

        self.cavity_type = cavity_type
        self.scaling = scaling
        self.n_angular = n_angular

        # Derived quantities
        # f(eps) = (eps - 1) / eps  — CPCM screening factor
        if self.epsilon > 1.0:
            self.f_epsilon = (self.epsilon - 1.0) / self.epsilon
        else:
            self.f_epsilon = 0.0  # Vacuum: no screening

        # Cached cavity (built on first use or explicit call)
        self._cavity: Optional[CavitySurface] = None
        self._cavity_molecule_id: Optional[int] = None

    @property
    def screening_factor(self) -> float:
        """CPCM screening factor f(eps) = (eps-1)/eps."""
        return self.f_epsilon

    # -----------------------------------------------------------------------
    # Cavity Construction
    # -----------------------------------------------------------------------

    def build_cavity(
        self,
        molecule: Molecule,
        verbose: bool = False,
    ) -> CavitySurface:
        """
        Construct molecular cavity as a solvent-excluded surface.

        The cavity is built from atom-centered spheres using scaled van der
        Waals radii. Surface points are generated on each sphere using angular
        quadrature, then points falling inside neighboring spheres are removed
        to form the solvent-excluded surface.

        Each surviving tessera has:
            - position: on the atomic sphere surface
            - normal: outward-pointing unit vector from atom center
            - area: angular weight * R^2 (sphere surface element)

        Args:
            molecule: Molecule object with atoms in Bohr coordinates
            verbose:  Print cavity construction details

        Returns:
            CavitySurface with tessera positions, normals, areas, atom indices
        """
        atoms = molecule.atoms
        n_atoms = len(atoms)

        if verbose:
            print(
                f"PCM: Building {self.cavity_type} cavity "
                f"(scaling={self.scaling:.2f}, n_ang={self.n_angular})"
            )

        # Get angular grid for surface tessellation
        ang_points, ang_weights = _lebedev_sphere(self.n_angular)
        n_ang = len(ang_weights)

        # Compute scaled vdW radii for each atom (in Bohr)
        radii_bohr = np.zeros(n_atoms)
        centers = np.zeros((n_atoms, 3))

        for i, (element, coords) in enumerate(atoms):
            r_angstrom = VDW_RADII_ANGSTROM.get(element, 1.70)  # Default C radius
            radii_bohr[i] = r_angstrom * ANGSTROM_TO_BOHR * self.scaling
            centers[i] = np.array(coords)

        if verbose:
            for i, (el, _) in enumerate(atoms):
                print(
                    f"  Atom {i} ({el}): R_vdw = {radii_bohr[i]:.4f} bohr "
                    f"({radii_bohr[i] * BOHR_TO_ANGSTROM:.4f} A)"
                )

        # Generate tesserae on each atomic sphere, removing interior points
        all_positions = []
        all_normals = []
        all_areas = []
        all_atom_idx = []

        for i in range(n_atoms):
            R_i = radii_bohr[i]
            center_i = centers[i]

            for k in range(n_ang):
                # Surface point on sphere i
                # normal = unit direction vector (already normalized, on unit sphere)
                normal_k = ang_points[k]
                point = center_i + R_i * normal_k

                # Solvent-excluded surface: discard if inside another sphere
                inside_other = False
                for j in range(n_atoms):
                    if j == i:
                        continue
                    dist_to_j = np.linalg.norm(point - centers[j])
                    if dist_to_j < radii_bohr[j]:
                        inside_other = True
                        break

                if inside_other:
                    continue

                # Area element: dA = R^2 * w_angular
                # The angular weight already contains the 4*pi normalization,
                # so the area element for a sphere of radius R is:
                # a_k = R^2 * w_k  (where sum(w_k) = 4*pi, so sum(a_k) = 4*pi*R^2)
                area_k = R_i**2 * ang_weights[k]

                all_positions.append(point)
                all_normals.append(normal_k)
                all_areas.append(area_k)
                all_atom_idx.append(i)

        if len(all_positions) == 0:
            raise RuntimeError(
                "PCM: Cavity construction produced zero tesserae. "
                "Check molecular geometry and vdW radii."
            )

        cavity = CavitySurface(
            positions=np.array(all_positions),
            normals=np.array(all_normals),
            areas=np.array(all_areas),
            atom_idx=np.array(all_atom_idx, dtype=int),
        )

        # Cache
        self._cavity = cavity
        self._cavity_molecule_id = id(molecule)

        if verbose:
            total_area = np.sum(cavity.areas)
            # Theoretical area of isolated spheres (no overlap)
            ideal_area = np.sum(4.0 * np.pi * radii_bohr**2)
            print(
                f"  Tesserae: {cavity.n_tesserae} (from {n_atoms * n_ang} candidates)"
            )
            print(
                f"  Total surface area: {total_area:.2f} bohr^2 "
                f"({total_area * BOHR_TO_ANGSTROM**2:.2f} A^2)"
            )
            print(f"  Isolated sphere area: {ideal_area:.2f} bohr^2")
            print(f"  SES / isolated ratio: {total_area / ideal_area:.3f}")

        return cavity

    def _get_cavity(self, molecule: Molecule, verbose: bool = False) -> CavitySurface:
        """Get or build cavity (with caching)."""
        if self._cavity is None or self._cavity_molecule_id != id(molecule):
            self.build_cavity(molecule, verbose=verbose)
        return self._cavity

    # -----------------------------------------------------------------------
    # Electrostatic Potential at Surface Points
    # -----------------------------------------------------------------------

    def _compute_nuclear_potential(
        self,
        molecule: Molecule,
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Compute electrostatic potential at surface points from nuclear charges.

        V_nuc(r) = sum_A Z_A / |r - R_A|

        Args:
            molecule:  Molecule with atomic coordinates (Bohr)
            positions: (n_tess, 3) surface point coordinates

        Returns:
            (n_tess,) array of nuclear electrostatic potential values (a.u.)
        """
        n_tess = len(positions)
        V = np.zeros(n_tess)

        for element, coords in molecule.atoms:
            Z = _Z_MAP.get(element, 0)
            if Z == 0:
                continue
            R_A = np.array(coords)
            # Vectorized distance computation
            diff = positions - R_A[np.newaxis, :]  # (n_tess, 3)
            dist = np.linalg.norm(diff, axis=1)  # (n_tess,)
            # Guard against singularity (point at nucleus — shouldn't happen
            # since cavity excludes nuclear region, but be safe)
            dist = np.maximum(dist, 1e-12)
            V += Z / dist

        return V

    def _compute_electronic_potential(
        self,
        molecule: Molecule,
        positions: np.ndarray,
        density_matrix: np.ndarray,
        basis: list,
    ) -> np.ndarray:
        """
        Compute electrostatic potential at surface points from electron density.

        V_elec(r) = -sum_{mu,nu} P_{mu,nu} * integral phi_mu(r') phi_nu(r') / |r-r'| dr'

        This requires one-electron nuclear-attraction-type integrals centered
        at each surface point. For computational efficiency, this uses the
        Mulliken charge approximation in the first implementation:

        V_elec(r) approx -sum_A q_A^Mulliken / |r - R_A|

        where q_A^Mulliken = sum_{mu in A} (P * S)_{mu,mu} is the Mulliken
        gross atomic population on atom A.

        Args:
            molecule:       Molecule object
            positions:      (n_tess, 3) surface point coordinates
            density_matrix: (N, N) AO density matrix P
            basis:          List of contracted Gaussian basis functions

        Returns:
            (n_tess,) array of electronic electrostatic potential (a.u.)
        """
        n_tess = len(positions)

        # ------------------------------------------------------------------
        # Full integral-based electronic potential
        # ------------------------------------------------------------------
        # For each surface point r_s, we need:
        #   V_elec(r_s) = -sum_{mu,nu} P_{mu,nu} * (mu nu | r_s)
        # where (mu nu | r_s) = integral phi_mu(r') phi_nu(r') / |r' - r_s| dr'
        #
        # This is equivalent to a nuclear attraction integral with a unit
        # positive charge at r_s. We use the integral engine for this.
        # ------------------------------------------------------------------

        try:
            try:
                from . import integrals as ints
            except ImportError:
                import integrals as ints

            N = density_matrix.shape[0]
            V_elec = np.zeros(n_tess)

            for s in range(n_tess):
                r_s = positions[s]
                # Build the one-electron potential matrix for a unit charge at r_s
                V_mat = np.zeros((N, N))
                for mu in range(N):
                    for nu in range(mu + 1):
                        val = 0.0
                        for p_mu in basis[mu].primitives:
                            for p_nu in basis[nu].primitives:
                                val += ints.nuclear_attraction(p_mu, p_nu, r_s, 1.0)
                        V_mat[mu, nu] = val
                        V_mat[nu, mu] = val

                # V_elec(r_s) = Tr(P * V_mat)
                # The nuclear_attraction integral already includes a -Z factor:
                #   V_mat[mu,nu] = nuclear_attraction(pmu, pnu, r_s, Z=1)
                #                = -integral phi_mu(r') phi_nu(r') / |r' - r_s| dr'
                # The ESP from electron density is:
                #   V_elec(r_s) = -sum P_{mu,nu} * integral(phi_mu phi_nu / |r'-r_s|)
                #               = -sum P_{mu,nu} * (-V_mat[mu,nu])
                #               = +Tr(P * V_mat)   [which is negative, as expected]
                V_elec[s] = np.sum(density_matrix * V_mat)

            return V_elec

        except (ImportError, AttributeError):
            # Fallback: Mulliken charge approximation
            return self._compute_electronic_potential_mulliken(
                molecule, positions, density_matrix, basis
            )

    def _compute_electronic_potential_mulliken(
        self,
        molecule: Molecule,
        positions: np.ndarray,
        density_matrix: np.ndarray,
        basis: list,
    ) -> np.ndarray:
        """
        Approximate electronic potential using Mulliken atomic charges.

        V_elec(r) = -sum_A q_A / |r - R_A|

        where q_A = sum_{mu on A} (P * S)_{mu,mu} is the Mulliken
        gross atomic population on atom A.

        This is faster than the full integral approach but less accurate.
        """
        n_tess = len(positions)
        N = density_matrix.shape[0]

        # Build overlap matrix S
        try:
            try:
                from . import integrals as ints
            except ImportError:
                import integrals as ints

            S = np.zeros((N, N))
            for mu in range(N):
                for nu in range(mu + 1):
                    val = 0.0
                    for p_mu in basis[mu].primitives:
                        for p_nu in basis[nu].primitives:
                            val += ints.overlap(p_mu, p_nu)
                    S[mu, nu] = val
                    S[nu, mu] = val

        except (ImportError, AttributeError):
            # If integrals not available, use identity (crude approximation)
            S = np.eye(N)

        # PS product
        PS = density_matrix @ S

        # Mulliken populations: q_A = sum_{mu on A} PS_{mu,mu}
        # Build basis-to-atom mapping
        atom_pops = np.zeros(len(molecule.atoms))
        mu_idx = 0
        for a_idx, (element, _) in enumerate(molecule.atoms):
            # Count basis functions centered on this atom
            # We need to know how many basis functions are on each atom.
            # In STO-3G: H=1, He=1, Li-Ne=5 (1s + 2s + 2px + 2py + 2pz)
            # In 6-31G: H=2, He=2, C-F=9, etc.
            # Use the basis set's atom assignment from integrals module.
            pass

        # Simpler approach: use basis function origin to assign to atoms
        atom_centers = np.array([coords for _, coords in molecule.atoms])
        basis_atom_map = np.zeros(N, dtype=int)
        for mu in range(N):
            if hasattr(basis[mu], "primitives") and len(basis[mu].primitives) > 0:
                origin = np.array(basis[mu].primitives[0].origin)
                dists = np.linalg.norm(atom_centers - origin[np.newaxis, :], axis=1)
                basis_atom_map[mu] = np.argmin(dists)
            else:
                basis_atom_map[mu] = 0

        # Compute Mulliken populations
        for mu in range(N):
            a_idx = basis_atom_map[mu]
            atom_pops[a_idx] += PS[mu, mu]

        # Electronic potential from Mulliken charges
        V_elec = np.zeros(n_tess)
        for a_idx, (element, coords) in enumerate(molecule.atoms):
            q_A = atom_pops[a_idx]  # Electron population (positive number)
            if abs(q_A) < 1e-15:
                continue
            R_A = np.array(coords)
            diff = positions - R_A[np.newaxis, :]
            dist = np.linalg.norm(diff, axis=1)
            dist = np.maximum(dist, 1e-12)
            V_elec -= q_A / dist  # Negative: electrons are negative charge

        return V_elec

    def _compute_total_potential(
        self,
        molecule: Molecule,
        positions: np.ndarray,
        solver=None,
    ) -> np.ndarray:
        """
        Compute total electrostatic potential at surface points.

        V(r) = V_nuc(r) + V_elec(r)

        If a converged HF solver is provided, uses the electron density.
        Otherwise, uses only nuclear charges (Born ion model).

        Args:
            molecule:  Molecule object
            positions: (n_tess, 3) surface point coordinates
            solver:    Converged HartreeFockSolver (optional)

        Returns:
            (n_tess,) total electrostatic potential
        """
        # Nuclear contribution (always present)
        V_nuc = self._compute_nuclear_potential(molecule, positions)

        # Electronic contribution (if HF solver available)
        if solver is not None and hasattr(solver, "P") and hasattr(solver, "basis"):
            V_elec = self._compute_electronic_potential(
                molecule, positions, solver.P, solver.basis
            )
        else:
            V_elec = np.zeros_like(V_nuc)

        return V_nuc + V_elec

    # -----------------------------------------------------------------------
    # Coulomb Matrix and Surface Charges
    # -----------------------------------------------------------------------

    def _build_coulomb_matrix(self, cavity: CavitySurface) -> np.ndarray:
        """
        Build the Coulomb interaction matrix D between surface tesserae.

        Off-diagonal: D[i,j] = 1 / |r_i - r_j|
        Diagonal:     D[i,i] = 1.07 * sqrt(4*pi / a_i)

        The diagonal self-interaction term follows the prescription of
        Cossi, Rega, Scalmani, Barone, JCC 24, 669 (2003), which accounts
        for the finite size of each tessera.

        The factor 1.07 is an empirical correction that improves agreement
        with exact dielectric boundary conditions for molecular-shaped cavities.

        Args:
            cavity: CavitySurface with tessera positions and areas

        Returns:
            (n_tess, n_tess) symmetric positive-definite Coulomb matrix
        """
        n = cavity.n_tesserae
        D = np.zeros((n, n))

        positions = cavity.positions
        areas = cavity.areas

        # Off-diagonal: Coulomb interaction 1/|r_i - r_j|
        # Use broadcasting for vectorized pairwise distances
        # diff[i,j] = r_i - r_j, shape (n, n, 3)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)  # (n, n)

        # Avoid division by zero on diagonal (will be overwritten)
        dist_safe = np.where(dist > 1e-15, dist, 1.0)
        D = 1.0 / dist_safe

        # Diagonal: self-interaction term
        # D[i,i] = 1.07 * sqrt(4*pi / a_i)
        # Ref: Cossi et al., JCC 24, 669 (2003), Eq. 9
        diag_values = 1.07 * np.sqrt(4.0 * np.pi / areas)
        np.fill_diagonal(D, diag_values)

        return D

    def _solve_surface_charges(
        self,
        molecule: Molecule,
        solver=None,
        cavity: Optional[CavitySurface] = None,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, CavitySurface]:
        """
        Solve the CPCM equation for apparent surface charges.

        q = -f(eps) * D^{-1} * V

        where:
            f(eps) = (eps - 1) / eps
            D = Coulomb interaction matrix
            V = electrostatic potential at tesserae

        Also returns the potential V for energy computation.

        Args:
            molecule: Molecule object
            solver:   Converged HF solver (optional, for electron density)
            cavity:   Pre-built cavity (optional, will build if None)
            verbose:  Print details

        Returns:
            Tuple of (q, V, cavity) where:
                q: (n_tess,) apparent surface charges
                V: (n_tess,) electrostatic potential at tesserae
                cavity: CavitySurface used
        """
        # Get cavity
        if cavity is None:
            cavity = self._get_cavity(molecule, verbose=verbose)

        # Special case: vacuum (no solvation)
        if self.f_epsilon == 0.0:
            n = cavity.n_tesserae
            return np.zeros(n), np.zeros(n), cavity

        if verbose:
            print(f"PCM: Solving CPCM equation ({cavity.n_tesserae} tesserae)")
            print(f"     Solvent: {self.solvent_name} (eps={self.epsilon:.2f})")
            print(f"     f(eps) = {self.f_epsilon:.6f}")

        # Compute electrostatic potential at tesserae
        V = self._compute_total_potential(molecule, cavity.positions, solver)

        if verbose:
            print(f"     |V|_max = {np.max(np.abs(V)):.6f} a.u.")
            print(f"     |V|_rms = {np.sqrt(np.mean(V**2)):.6f} a.u.")

        # Build Coulomb matrix
        D = self._build_coulomb_matrix(cavity)

        # Solve: q = -f(eps) * D^{-1} * V
        # Use np.linalg.solve for numerical stability (avoids explicit inverse)
        # D * q_bare = V  =>  q_bare = D^{-1} * V
        try:
            q_bare = np.linalg.solve(D, V)
        except np.linalg.LinAlgError:
            import warnings

            warnings.warn(
                "PCM: Coulomb matrix is singular. Using pseudoinverse.",
                stacklevel=2,
            )
            D_pinv = np.linalg.pinv(D)
            q_bare = D_pinv @ V

        q = -self.f_epsilon * q_bare

        if verbose:
            Q_total = np.sum(q)
            # Net solute charge = nuclear charge - electron count
            # With full density: net charge equals formal molecular charge
            # Without density: net charge is total nuclear charge Z_total
            Z_total = sum(_Z_MAP.get(el, 0) for el, _ in molecule.atoms)
            if solver is not None and hasattr(solver, "P"):
                N_elec = np.trace(solver.P @ np.eye(solver.P.shape[0]))
                # With overlap: N_elec = Tr(P*S), but for diagnostics Tr(P) is approximate
                Q_solute_eff = molecule.charge  # Formal charge is exact
            else:
                Q_solute_eff = Z_total  # Nuclear-only model
            Q_expected = -self.f_epsilon * Q_solute_eff
            print(f"     Total surface charge: {Q_total:+.6f} e")
            print(
                f"     Expected (Gauss):     {Q_expected:+.6f} e "
                f"(-f*Q_solute, Q_solute={Q_solute_eff:+.2f})"
            )
            if abs(Q_solute_eff) > 0.01:
                print(f"     Deviation:            {abs(Q_total - Q_expected):.2e} e")

        return q, V, cavity

    # -----------------------------------------------------------------------
    # Solvation Energy
    # -----------------------------------------------------------------------

    def compute_solvation_energy(
        self,
        molecule: Molecule,
        solver=None,
        verbose: bool = True,
    ) -> float:
        """
        Compute the electrostatic solvation free energy (CPCM).

        E_solv = 0.5 * sum(q_i * V_i)

        This is the electrostatic component of the solvation free energy,
        computed as a post-SCF correction (non-self-consistent).

        Args:
            molecule: Molecule object with atoms in Bohr
            solver:   Converged HartreeFockSolver (optional)
                      If provided, uses the full electron density for V.
                      If None, uses nuclear charges only (Born ion model).
            verbose:  Print energy decomposition

        Returns:
            Solvation energy in Hartree (negative = stabilizing)
        """
        # Solve for surface charges
        q, V, cavity = self._solve_surface_charges(
            molecule, solver=solver, verbose=verbose
        )

        # Solvation energy: E_solv = 0.5 * q . V
        E_solv = 0.5 * np.dot(q, V)

        if verbose:
            print(f"\n{'=' * 55}")
            print(f"  CPCM Solvation Energy")
            print(f"{'=' * 55}")
            print(f"  Solvent:          {self.solvent_name} (eps={self.epsilon:.2f})")
            print(f"  Screening factor: {self.f_epsilon:.6f}")
            print(f"  Tesserae:         {cavity.n_tesserae}")
            print(f"  E_solv = {E_solv:16.10f} Hartree")
            print(f"         = {E_solv * HARTREE_TO_EV:16.10f} eV")
            print(f"         = {E_solv * HARTREE_TO_KCAL:16.6f} kcal/mol")
            print(f"{'=' * 55}")

        # Store results
        self.E_solv = E_solv
        self.surface_charges = q
        self.surface_potential = V
        self._cavity = cavity

        return E_solv

    def compute_solvated_energy(
        self,
        molecule: Molecule,
        solver=None,
        verbose: bool = True,
    ) -> float:
        """
        Compute total energy in solution (gas-phase + solvation correction).

        E_solution = E_gas + E_solv

        This is the post-SCF (non-self-consistent) approach:
        1. Run gas-phase HF to get E_gas and electron density
        2. Compute CPCM solvation correction E_solv
        3. Add them together

        For a self-consistent approach (PCM embedded in SCF), the solvation
        potential should be added to the Fock matrix at each iteration.
        That requires modifying the SCF loop and is planned for a future version.

        Args:
            molecule: Molecule object with atoms in Bohr
            solver:   Converged HartreeFockSolver with E_elec attribute
                      (E_elec stores total HF energy per convention)
            verbose:  Print energy decomposition

        Returns:
            Total energy in solution (Hartree)
        """
        # Get gas-phase energy
        E_gas = 0.0
        if solver is not None and hasattr(solver, "E_elec"):
            E_gas = solver.E_elec  # Convention: E_elec IS E_total
        elif solver is not None:
            import warnings

            warnings.warn(
                "PCM: Solver has no E_elec attribute. "
                "Run compute_energy() first. Using E_gas = 0.",
                stacklevel=2,
            )

        # Compute solvation correction
        E_solv = self.compute_solvation_energy(molecule, solver, verbose=verbose)

        E_solution = E_gas + E_solv

        if verbose:
            print(f"\n  E(gas)      = {E_gas:16.10f} Hartree")
            print(f"  E(solv)     = {E_solv:16.10f} Hartree")
            print(f"  E(solution) = {E_solution:16.10f} Hartree")
            print(f"  dE(solv)    = {E_solv * HARTREE_TO_KCAL:+.4f} kcal/mol")

        self.E_gas = E_gas
        self.E_solution = E_solution

        return E_solution

    # -----------------------------------------------------------------------
    # Analysis Utilities
    # -----------------------------------------------------------------------

    def compute_apparent_charges_by_atom(
        self,
        molecule: Molecule,
        solver=None,
    ) -> Dict[int, float]:
        """
        Partition surface charges by parent atom.

        Returns a dict mapping atom index to total apparent surface charge
        on that atom's portion of the cavity.

        Args:
            molecule: Molecule object
            solver:   Converged HF solver (optional)

        Returns:
            Dict[atom_index, total_charge_on_atom]
        """
        if not hasattr(self, "surface_charges") or self.surface_charges is None:
            self.compute_solvation_energy(molecule, solver, verbose=False)

        q = self.surface_charges
        cavity = self._cavity

        charges_by_atom: Dict[int, float] = {}
        for i in range(cavity.n_tesserae):
            a = cavity.atom_idx[i]
            charges_by_atom[a] = charges_by_atom.get(a, 0.0) + q[i]

        return charges_by_atom

    def compute_solvation_energy_decomposition(
        self,
        molecule: Molecule,
        solver=None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Decompose solvation energy by atom contribution.

        E_solv = sum_A E_solv^A where E_solv^A = 0.5 * sum_{i in A} q_i * V_i

        Args:
            molecule: Molecule object
            solver:   Converged HF solver (optional)
            verbose:  Print decomposition table

        Returns:
            Dict with per-atom and total solvation energies (Hartree)
        """
        q, V, cavity = self._solve_surface_charges(
            molecule, solver=solver, verbose=False
        )

        result: Dict[str, float] = {}
        total = 0.0

        for a_idx, (element, _) in enumerate(molecule.atoms):
            mask = cavity.atom_idx == a_idx
            E_a = 0.5 * np.dot(q[mask], V[mask])
            key = f"atom_{a_idx}_{element}"
            result[key] = E_a
            total += E_a

        result["total"] = total

        if verbose:
            print(f"\n  Solvation Energy Decomposition by Atom")
            print(
                f"  {'Atom':>10s}  {'E_solv (Hartree)':>18s}  {'E_solv (kcal/mol)':>18s}"
            )
            print(f"  {'-' * 50}")
            for a_idx, (element, _) in enumerate(molecule.atoms):
                key = f"atom_{a_idx}_{element}"
                E_a = result[key]
                print(
                    f"  {a_idx:>4d} {element:>4s}  {E_a:18.10f}  {E_a * HARTREE_TO_KCAL:18.6f}"
                )
            print(f"  {'-' * 50}")
            print(f"  {'Total':>10s}  {total:18.10f}  {total * HARTREE_TO_KCAL:18.6f}")

        return result

    def compare_solvents(
        self,
        molecule: Molecule,
        solver=None,
        solvents: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Compare solvation energies across multiple solvents.

        Useful for studying solvent effects on molecular energetics.

        Args:
            molecule: Molecule object
            solver:   Converged HF solver (optional)
            solvents: List of solvent names (default: all)
            verbose:  Print comparison table

        Returns:
            Dict mapping solvent name to solvation energy (Hartree)
        """
        if solvents is None:
            solvents = [s for s in SOLVENTS if s != "vacuum"]

        results: Dict[str, float] = {}

        # Save current state
        orig_epsilon = self.epsilon
        orig_name = self.solvent_name
        orig_f = self.f_epsilon

        for solvent_name in solvents:
            if solvent_name.lower() not in SOLVENTS:
                continue

            # Temporarily change solvent
            self.epsilon = SOLVENTS[solvent_name.lower()]["epsilon"]
            self.solvent_name = SOLVENTS[solvent_name.lower()]["name"]
            if self.epsilon > 1.0:
                self.f_epsilon = (self.epsilon - 1.0) / self.epsilon
            else:
                self.f_epsilon = 0.0

            E_solv = self.compute_solvation_energy(molecule, solver, verbose=False)
            results[solvent_name] = E_solv

        # Restore original state
        self.epsilon = orig_epsilon
        self.solvent_name = orig_name
        self.f_epsilon = orig_f

        if verbose:
            print(f"\n  Solvation Energy Comparison")
            print(
                f"  {'Solvent':>15s}  {'eps':>8s}  "
                f"{'E_solv (Ha)':>14s}  {'E_solv (kcal/mol)':>18s}"
            )
            print(f"  {'-' * 60}")
            for name in solvents:
                if name in results:
                    eps = SOLVENTS[name.lower()]["epsilon"]
                    E = results[name]
                    print(
                        f"  {name:>15s}  {eps:8.2f}  "
                        f"{E:14.8f}  {E * HARTREE_TO_KCAL:18.6f}"
                    )
            print(f"  {'-' * 60}")

        return results

    # -----------------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PCMSolver(solvent='{self.solvent_name}', "
            f"eps={self.epsilon:.2f}, "
            f"f(eps)={self.f_epsilon:.6f}, "
            f"cavity='{self.cavity_type}', "
            f"scaling={self.scaling:.2f})"
        )
