"""
Geometry Optimizer Module
Provides routines to optimize molecular geometry using Hartree-Fock energy gradients.

Methods:
  - Steepest Descent (legacy, `optimize()`)
  - BFGS quasi-Newton (default, `optimize_bfgs()`)
  - L-BFGS limited-memory quasi-Newton (`optimize_lbfgs()`)

All coordinates are in Bohr. Only numpy is used — no external optimization libraries.
"""

import numpy as np
from copy import deepcopy

# Support both relative (package) and absolute (standalone) imports
try:
    from .solver import HartreeFockSolver
    from .molecule import Molecule
except ImportError:
    from solver import HartreeFockSolver
    from molecule import Molecule

# Atomic masses in amu for center-of-mass projection (rows 1-3)
_ATOMIC_MASS = {
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


def _atoms_to_flat(atoms):
    """Extract a flat numpy array of coordinates from molecule.atoms."""
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


def _project_translations(grad_flat, atoms):
    """
    Project out center-of-mass translation from the gradient.

    This ensures the optimizer does not move the molecular center of mass.
    The projection is mass-weighted: for each Cartesian direction d,
    we subtract the mass-weighted average force so that
    sum_A m_A * g_A_d = 0 after projection.
    """
    n_atoms = len(atoms)
    masses = np.array([_ATOMIC_MASS.get(el, 12.0) for el, _ in atoms])
    total_mass = np.sum(masses)

    g = grad_flat.copy()
    for d in range(3):  # x, y, z
        # Indices for this Cartesian direction across all atoms
        indices = np.arange(d, 3 * n_atoms, 3)
        # Mass-weighted average force in direction d
        weighted_avg = np.sum(masses * g[indices]) / total_mass
        g[indices] -= weighted_avg

    return g


def _project_rotations(grad_flat, atoms):
    """
    Project out infinitesimal rotations from the gradient for non-linear molecules.

    For a molecule with N_atoms > 2 and non-collinear atoms, there are 3 rotational
    degrees of freedom. We project out the component of the gradient along each
    rotational eigenvector (mass-weighted cross products with position).

    For linear molecules (or N_atoms <= 2), only 2 rotational DOFs exist, but
    the Gram-Schmidt procedure naturally handles rank deficiency.
    """
    n_atoms = len(atoms)
    if n_atoms < 2:
        return grad_flat

    masses = np.array([_ATOMIC_MASS.get(el, 12.0) for el, _ in atoms])
    total_mass = np.sum(masses)

    # Compute center of mass
    com = np.zeros(3)
    for i, (el, (x, y, z)) in enumerate(atoms):
        com += masses[i] * np.array([x, y, z])
    com /= total_mass

    # Build rotation vectors: R_d = m_A * (e_d x (r_A - com))
    # These are 3N-dimensional vectors corresponding to infinitesimal rotations
    # around each of the 3 Cartesian axes.
    rot_vectors = []
    for d in range(3):
        e_d = np.zeros(3)
        e_d[d] = 1.0
        vec = np.zeros(3 * n_atoms)
        for i, (el, (x, y, z)) in enumerate(atoms):
            r = np.array([x, y, z]) - com
            cross = np.cross(e_d, r)
            vec[3 * i : 3 * i + 3] = np.sqrt(masses[i]) * cross
        rot_vectors.append(vec)

    # Gram-Schmidt orthonormalize the rotation vectors
    # (handles linear molecules where one vector is zero)
    ortho = []
    for v in rot_vectors:
        for u in ortho:
            v = v - np.dot(v, u) * u
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            ortho.append(v / norm)

    # Project out rotational components from the gradient
    g = grad_flat.copy()
    for u in ortho:
        g -= np.dot(g, u) * u

    return g


class GeometryOptimizer:
    """
    Molecular geometry optimizer using quasi-Newton methods (BFGS/L-BFGS).

    Uses numerical gradients via central finite difference and
    projects out translational/rotational degrees of freedom.
    Supports Armijo line search and Hessian updating.
    """

    def __init__(self, solver: HartreeFockSolver):
        self.solver = solver
        self.history = []

    def compute_gradient_numerical(self, molecule: Molecule, step_size=0.005):
        """
        Computes the nuclear gradient via Central Finite Difference.
        Grad_A = [E(R_A + h) - E(R_A - h)] / (2h)
        """
        atoms = molecule.atoms
        gradients = []

        print(f"  Computing Numerical Gradient (step={step_size})...")

        for i in range(len(atoms)):
            atom_grad = []
            element, original_pos = atoms[i]
            original_pos = np.array(original_pos)

            # For each Cartesian component (x, y, z)
            for axis in range(3):
                # Shift +h
                pos_plus = original_pos.copy()
                pos_plus[axis] += step_size

                mol_plus = deepcopy(molecule)
                mol_plus.atoms[i] = (element, tuple(pos_plus))

                e_plus, _ = self.solver.compute_energy(
                    mol_plus, max_iter=40, tolerance=1e-6
                )

                # Shift -h
                pos_minus = original_pos.copy()
                pos_minus[axis] -= step_size

                mol_minus = deepcopy(molecule)
                mol_minus.atoms[i] = (element, tuple(pos_minus))

                e_minus, _ = self.solver.compute_energy(
                    mol_minus, max_iter=40, tolerance=1e-6
                )

                # Central Difference
                grad_component = (e_plus - e_minus) / (2.0 * step_size)
                atom_grad.append(grad_component)

            gradients.append(tuple(atom_grad))

        return gradients

    def compute_gradient_analytical(self, molecule: Molecule):
        """
        Computes the nuclear gradient using Analytical Derivatives.
        """
        print("  Computing Analytical Gradient...")
        return self.solver.compute_gradient(molecule)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _get_gradient(self, molecule, method):
        """Compute gradient via the requested method and return as flat array."""
        if method == "analytical":
            grads = self.compute_gradient_analytical(molecule)
        else:
            grads = self.compute_gradient_numerical(molecule)
        return _gradient_to_flat(grads)

    def _make_molecule(self, flat_coords, template_mol):
        """Build a new Molecule from flat coordinates, copying metadata."""
        new_atoms = _flat_to_atoms(flat_coords, template_mol.atoms)
        mol = deepcopy(template_mol)
        mol.atoms = new_atoms
        return mol

    def _energy(self, molecule):
        """Compute total energy (first element of the tuple)."""
        e_total, _ = self.solver.compute_energy(molecule)
        return e_total

    @staticmethod
    def _print_step_info(
        step, max_steps, energy, rms_force, max_force, atoms, gradients
    ):
        """Print formatted convergence information for one step."""
        print(f"\n[Step {step}/{max_steps}]")
        print(f"  Energy    : {energy:.10f} Ha")
        print(f"  RMS Force : {rms_force:.8f} Ha/Bohr")
        print(f"  Max Force : {max_force:.8f} Ha/Bohr")
        n_atoms = len(atoms)
        if n_atoms <= 10:
            for i in range(n_atoms):
                el = atoms[i][0]
                gx = gradients[3 * i]
                gy = gradients[3 * i + 1]
                gz = gradients[3 * i + 2]
                print(f"    Atom {i} ({el}): [{gx:+.8f}, {gy:+.8f}, {gz:+.8f}]")

    # ------------------------------------------------------------------
    #  Steepest Descent (legacy)
    # ------------------------------------------------------------------

    def optimize(
        self,
        molecule: Molecule,
        max_steps=20,
        learning_rate=0.5,
        tolerance=1e-3,
        method="analytical",
    ):
        """
        Performs geometry optimization using Steepest Descent.
        R_new = R_old - learning_rate * Gradient

        Args:
            method: 'analytical' (default) or 'numerical'

        Kept for backward compatibility. For production use, prefer
        optimize_bfgs() or optimize_lbfgs().
        """
        print("========================================")
        print("Geometry Optimization Initiated")
        print("  Method: Steepest Descent (legacy)")
        print("========================================")

        current_mol = deepcopy(molecule)

        for step in range(max_steps):
            print(f"\n[Step {step + 1}/{max_steps}]")

            # 1. Compute Energy
            energy, _ = self.solver.compute_energy(current_mol)
            self.history.append(energy)
            print(f"Energy: {energy:.6f} Ha")

            # 2. Compute Gradient
            if method == "analytical":
                gradients = self.compute_gradient_analytical(current_mol)
            else:
                gradients = self.compute_gradient_numerical(current_mol)

            # 3. Check Convergence (RMS Force)
            grad_array = np.array(gradients)
            rms_force = np.sqrt(np.mean(grad_array**2))
            max_force = np.max(np.abs(grad_array))

            print(f"Forces (RMS: {rms_force:.6f}, Max: {max_force:.6f}):")
            for i, g in enumerate(gradients):
                print(f"  Atom {i} ({current_mol.atoms[i][0]}): {g}")

            if rms_force < tolerance:
                print("========================================")
                print("Geometry Converged!")
                print("========================================")
                return current_mol, self.history

            # 4. Update Coordinates
            new_atoms = []
            for i, (el, pos) in enumerate(current_mol.atoms):
                g = np.array(gradients[i])
                new_pos = np.array(pos) - learning_rate * g
                new_atoms.append((el, tuple(new_pos)))

            current_mol.atoms = new_atoms

        print("Warning: Maximum steps reached without full convergence.")
        return current_mol, self.history

    # ------------------------------------------------------------------
    #  BFGS Quasi-Newton Optimizer
    # ------------------------------------------------------------------

    def optimize_bfgs(
        self,
        molecule: Molecule,
        max_steps=50,
        tolerance=1e-4,
        method="analytical",
        initial_step=1.0,
        project_translations=True,
        project_rotations=True,
    ):
        """
        BFGS (Broyden-Fletcher-Goldfarb-Shanno) quasi-Newton geometry optimizer.

        Uses an inverse Hessian approximation that is iteratively updated via
        the standard BFGS formula. A backtracking line search along the Newton
        direction ensures sufficient energy decrease (Armijo condition).

        Convergence criterion: max |gradient component| < tolerance.

        Args:
            molecule:             Input molecular geometry (Bohr).
            max_steps:            Maximum optimisation iterations.
            tolerance:            Convergence threshold on max force (Ha/Bohr).
            method:               'analytical' (default) or 'numerical' gradient.
            initial_step:         Initial step length for the line search.
            project_translations: Remove centre-of-mass drift from gradient.
            project_rotations:    Remove rigid rotations from gradient.

        Returns:
            (optimized_molecule, energy_history)
        """
        print("========================================")
        print("Geometry Optimization — BFGS")
        print(f"  Convergence: max force < {tolerance:.1e} Ha/Bohr")
        print(f"  Max steps  : {max_steps}")
        print(f"  Gradient   : {method}")
        print("========================================")

        self.history = []
        geom_history = []

        current_mol = deepcopy(molecule)
        n_coords = 3 * len(current_mol.atoms)

        # --- Initial energy and gradient ---
        energy = self._energy(current_mol)
        grad = self._get_gradient(current_mol, method)
        if project_translations:
            grad = _project_translations(grad, current_mol.atoms)
        if project_rotations:
            grad = _project_rotations(grad, current_mol.atoms)

        self.history.append(energy)
        geom_history.append(deepcopy(current_mol.atoms))

        # Inverse Hessian approximation — start with identity
        H_inv = np.eye(n_coords)

        for step in range(1, max_steps + 1):
            # --- Convergence check ---
            rms_force = np.sqrt(np.mean(grad**2))
            max_force = np.max(np.abs(grad))
            self._print_step_info(
                step, max_steps, energy, rms_force, max_force, current_mol.atoms, grad
            )

            if max_force < tolerance:
                print("\n========================================")
                print("BFGS Converged!")
                print(f"  Final energy : {energy:.10f} Ha")
                print(f"  Max force    : {max_force:.2e} Ha/Bohr")
                print(f"  Steps        : {step}")
                print("========================================")
                return current_mol, self.history

            # --- Search direction: p = -H_inv @ g ---
            p = -H_inv @ grad

            # --- Backtracking line search (Armijo) ---
            alpha = initial_step
            c1 = 1e-4  # Armijo parameter
            rho_ls = 0.5  # step reduction factor
            max_ls_iter = 20
            directional_deriv = np.dot(grad, p)

            # Safeguard: if p is not a descent direction (can happen if H_inv
            # becomes non-positive-definite due to numerical noise), reset to
            # steepest descent.
            if directional_deriv > 0:
                print("  [BFGS] Search direction is not descent — resetting H_inv")
                H_inv = np.eye(n_coords)
                p = -grad
                directional_deriv = np.dot(grad, p)

            x_old = _atoms_to_flat(current_mol.atoms)

            for ls_iter in range(max_ls_iter):
                x_new = x_old + alpha * p
                trial_mol = self._make_molecule(x_new, current_mol)
                trial_energy = self._energy(trial_mol)

                # Armijo sufficient decrease condition
                if trial_energy <= energy + c1 * alpha * directional_deriv:
                    break
                alpha *= rho_ls
            else:
                # If line search exhausted, take a small steepest-descent step
                print("  [BFGS] Line search failed — taking small SD step")
                alpha = 0.01
                x_new = x_old - alpha * grad
                trial_mol = self._make_molecule(x_new, current_mol)
                trial_energy = self._energy(trial_mol)

            # --- Compute new gradient ---
            new_grad = self._get_gradient(trial_mol, method)
            if project_translations:
                new_grad = _project_translations(new_grad, trial_mol.atoms)
            if project_rotations:
                new_grad = _project_rotations(new_grad, trial_mol.atoms)

            # --- BFGS inverse Hessian update ---
            s = x_new - x_old  # step vector
            y = new_grad - grad  # gradient change

            sy = np.dot(s, y)
            if sy > 1e-12:
                rho = 1.0 / sy
                I = np.eye(n_coords)
                # H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
                sy_outer = rho * np.outer(s, y)
                ys_outer = rho * np.outer(y, s)
                ss_outer = rho * np.outer(s, s)
                H_inv = (I - sy_outer) @ H_inv @ (I - ys_outer) + ss_outer
            else:
                # Curvature condition violated — skip update (keep previous H_inv)
                print(
                    "  [BFGS] Curvature condition not met (s.y <= 0) — skipping H update"
                )

            # --- Update state for next iteration ---
            current_mol = trial_mol
            energy = trial_energy
            grad = new_grad
            self.history.append(energy)
            geom_history.append(deepcopy(current_mol.atoms))

        # --- Did not converge within max_steps ---
        rms_force = np.sqrt(np.mean(grad**2))
        max_force = np.max(np.abs(grad))
        print("\n========================================")
        print("Warning: BFGS did not converge within max steps.")
        print(f"  Final energy : {energy:.10f} Ha")
        print(f"  Max force    : {max_force:.2e} Ha/Bohr")
        print("========================================")
        return current_mol, self.history

    # ------------------------------------------------------------------
    #  L-BFGS (Limited-Memory BFGS) Optimizer
    # ------------------------------------------------------------------

    def optimize_lbfgs(
        self,
        molecule: Molecule,
        max_steps=100,
        tolerance=1e-4,
        method="analytical",
        initial_step=1.0,
        m=7,
        project_translations=True,
        project_rotations=True,
    ):
        """
        L-BFGS geometry optimizer for larger molecules.

        Instead of storing the full N x N inverse Hessian, L-BFGS stores the
        last *m* (s, y) vector pairs and reconstructs the search direction via
        the two-loop recursion of Nocedal (1980).

        Memory: O(m * N) instead of O(N^2) — practical for systems with
        hundreds of atoms.

        Args:
            molecule:             Input molecular geometry (Bohr).
            max_steps:            Maximum optimisation iterations.
            tolerance:            Convergence threshold on max force (Ha/Bohr).
            method:               'analytical' or 'numerical' gradient.
            initial_step:         Initial step length for the line search.
            m:                    Number of stored correction pairs (default 7).
            project_translations: Remove centre-of-mass drift from gradient.
            project_rotations:    Remove rigid rotations from gradient.

        Returns:
            (optimized_molecule, energy_history)
        """
        print("========================================")
        print("Geometry Optimization — L-BFGS")
        print(f"  Convergence : max force < {tolerance:.1e} Ha/Bohr")
        print(f"  Max steps   : {max_steps}")
        print(f"  History size: m = {m}")
        print(f"  Gradient    : {method}")
        print("========================================")

        self.history = []
        geom_history = []

        current_mol = deepcopy(molecule)
        n_coords = 3 * len(current_mol.atoms)

        # --- Initial energy and gradient ---
        energy = self._energy(current_mol)
        grad = self._get_gradient(current_mol, method)
        if project_translations:
            grad = _project_translations(grad, current_mol.atoms)
        if project_rotations:
            grad = _project_rotations(grad, current_mol.atoms)

        self.history.append(energy)
        geom_history.append(deepcopy(current_mol.atoms))

        # Storage for the last m correction pairs
        s_hist = []  # list of s_k vectors (position changes)
        y_hist = []  # list of y_k vectors (gradient changes)
        rho_hist = []  # list of rho_k = 1 / (y_k . s_k)

        for step in range(1, max_steps + 1):
            # --- Convergence check ---
            rms_force = np.sqrt(np.mean(grad**2))
            max_force = np.max(np.abs(grad))
            self._print_step_info(
                step, max_steps, energy, rms_force, max_force, current_mol.atoms, grad
            )

            if max_force < tolerance:
                print("\n========================================")
                print("L-BFGS Converged!")
                print(f"  Final energy : {energy:.10f} Ha")
                print(f"  Max force    : {max_force:.2e} Ha/Bohr")
                print(f"  Steps        : {step}")
                print("========================================")
                return current_mol, self.history

            # --- Two-loop recursion to compute search direction ---
            p = self._lbfgs_direction(grad, s_hist, y_hist, rho_hist)

            # --- Backtracking line search (Armijo) ---
            alpha = initial_step
            c1 = 1e-4
            rho_ls = 0.5
            max_ls_iter = 20
            directional_deriv = np.dot(grad, p)

            # If not a descent direction, fall back to steepest descent and
            # clear L-BFGS history.
            if directional_deriv > 0:
                print("  [L-BFGS] Search direction is not descent — resetting history")
                s_hist.clear()
                y_hist.clear()
                rho_hist.clear()
                p = -grad
                directional_deriv = np.dot(grad, p)

            x_old = _atoms_to_flat(current_mol.atoms)

            for ls_iter in range(max_ls_iter):
                x_new = x_old + alpha * p
                trial_mol = self._make_molecule(x_new, current_mol)
                trial_energy = self._energy(trial_mol)

                if trial_energy <= energy + c1 * alpha * directional_deriv:
                    break
                alpha *= rho_ls
            else:
                print("  [L-BFGS] Line search failed — taking small SD step")
                alpha = 0.01
                x_new = x_old - alpha * grad
                trial_mol = self._make_molecule(x_new, current_mol)
                trial_energy = self._energy(trial_mol)

            # --- Compute new gradient ---
            new_grad = self._get_gradient(trial_mol, method)
            if project_translations:
                new_grad = _project_translations(new_grad, trial_mol.atoms)
            if project_rotations:
                new_grad = _project_rotations(new_grad, trial_mol.atoms)

            # --- Store correction pair ---
            s = x_new - x_old
            y = new_grad - grad
            sy = np.dot(s, y)

            if sy > 1e-12:
                if len(s_hist) >= m:
                    s_hist.pop(0)
                    y_hist.pop(0)
                    rho_hist.pop(0)
                s_hist.append(s.copy())
                y_hist.append(y.copy())
                rho_hist.append(1.0 / sy)
            else:
                print("  [L-BFGS] Curvature condition not met — discarding pair")

            # --- Update state ---
            current_mol = trial_mol
            energy = trial_energy
            grad = new_grad
            self.history.append(energy)
            geom_history.append(deepcopy(current_mol.atoms))

        # --- Did not converge ---
        rms_force = np.sqrt(np.mean(grad**2))
        max_force = np.max(np.abs(grad))
        print("\n========================================")
        print("Warning: L-BFGS did not converge within max steps.")
        print(f"  Final energy : {energy:.10f} Ha")
        print(f"  Max force    : {max_force:.2e} Ha/Bohr")
        print("========================================")
        return current_mol, self.history

    @staticmethod
    def _lbfgs_direction(grad, s_hist, y_hist, rho_hist):
        """
        L-BFGS two-loop recursion (Nocedal 1980).

        Given the current gradient and the stored correction pairs
        {s_k, y_k, rho_k}, compute the search direction p = -H_k g
        without explicitly forming H_k.

        The initial Hessian scaling uses gamma_k = (s_{k-1} . y_{k-1}) /
        (y_{k-1} . y_{k-1}) to approximate curvature, which is the
        standard Shanno-Phua scaling.
        """
        q = grad.copy()
        k = len(s_hist)

        if k == 0:
            # No history yet — steepest descent
            return -q

        # --- Forward loop (most recent to oldest) ---
        alpha_arr = np.zeros(k)
        for i in range(k - 1, -1, -1):
            alpha_arr[i] = rho_hist[i] * np.dot(s_hist[i], q)
            q = q - alpha_arr[i] * y_hist[i]

        # --- Initial Hessian scaling (Shanno-Phua) ---
        # H_0 = gamma * I, where gamma = s_{k-1}.y_{k-1} / (y_{k-1}.y_{k-1})
        sy_last = np.dot(s_hist[-1], y_hist[-1])
        yy_last = np.dot(y_hist[-1], y_hist[-1])
        if yy_last > 1e-16:
            gamma = sy_last / yy_last
        else:
            gamma = 1.0
        r = gamma * q

        # --- Backward loop (oldest to most recent) ---
        for i in range(k):
            beta = rho_hist[i] * np.dot(y_hist[i], r)
            r = r + (alpha_arr[i] - beta) * s_hist[i]

        return -r

    # ------------------------------------------------------------------
    #  Default optimize entry point
    # ------------------------------------------------------------------

    def optimize_geometry(self, molecule: Molecule, method_opt="bfgs", **kwargs):
        """
        Unified entry point for geometry optimisation.

        Args:
            molecule:   Input Molecule.
            method_opt: 'bfgs' (default), 'lbfgs', or 'sd' (steepest descent).
            **kwargs:   Forwarded to the selected optimizer method.

        Returns:
            (optimized_molecule, energy_history)
        """
        dispatch = {
            "bfgs": self.optimize_bfgs,
            "lbfgs": self.optimize_lbfgs,
            "sd": self.optimize,
        }
        if method_opt not in dispatch:
            raise ValueError(
                f"Unknown optimization method '{method_opt}'. "
                f"Choose from: {list(dispatch.keys())}"
            )
        return dispatch[method_opt](molecule, **kwargs)
