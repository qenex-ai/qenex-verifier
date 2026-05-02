"""
CCSD Analytical Gradient Module (Numerical Implementation)
============================================================
Computes nuclear energy gradients at the CCSD level of theory
using central finite differences.

The numerical gradient is correct-by-construction:

    dE/dR_Ai = [E(R_Ai + h) - E(R_Ai - h)] / (2h)

where E = E_HF + E_corr(CCSD) is the total CCSD energy evaluated
at each displaced geometry by running a full HF + CCSD calculation.

Cost: 6 * N_atoms HF+CCSD single-point calculations per gradient
evaluation. This is expensive but avoids the complexity of the
analytical gradient (Z-vector equations, relaxed density matrices,
AO derivative integrals). Analytical gradients can be implemented
as a future optimization following Scheiner/Scuseria/Lee (1987).

Also provides a BFGS geometry optimizer driven by CCSD gradients,
enabling geometry optimization at the CCSD level.

All coordinates in Bohr. Energies in Hartree. Gradients in Hartree/Bohr.

References:
    Scheiner, Scuseria, Lee, J. Chem. Phys. 87, 5361 (1987)
    Salter, Trucks, Bartlett, J. Chem. Phys. 90, 1752 (1989)
"""

import numpy as np
from copy import deepcopy

# Support both package and direct imports
try:
    from .solver import HartreeFockSolver
    from .ccsd import CCSDSolver
    from .molecule import Molecule
except ImportError:
    from solver import HartreeFockSolver
    from ccsd import CCSDSolver
    from molecule import Molecule

__all__ = ["CCSDGradient"]


class CCSDGradient:
    """
    CCSD nuclear energy gradient via central finite differences.

    At each displaced geometry, a full HF + CCSD calculation is performed.
    The gradient is then:

        g_Ai = [E_CCSD(R_Ai + h) - E_CCSD(R_Ai - h)] / (2h)

    where A is the atom index and i is the Cartesian direction (x, y, z).

    Example usage::

        from molecule import Molecule
        from solver import HartreeFockSolver
        from ccsd import CCSDSolver
        from ccsd_gradient import CCSDGradient

        mol = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 1.4))], basis_name="sto-3g")
        hf = HartreeFockSolver()
        hf.compute_energy(mol, verbose=False)
        ccsd = CCSDSolver(convergence=1e-10)
        ccsd.solve(hf, mol, verbose=False)

        grad = CCSDGradient()
        g = grad.compute_gradient(mol, hf, ccsd, verbose=True)
        # g.shape == (2, 3) — gradient for each atom in each direction
    """

    def compute_gradient(
        self, molecule, hf_solver, ccsd_solver, step_size=0.005, verbose=False
    ):
        """
        Compute the CCSD nuclear energy gradient via central finite differences.

        For each atom A and Cartesian direction i (x, y, z):
            1. Displace atom A by +h in direction i → compute E_CCSD(+)
            2. Displace atom A by -h in direction i → compute E_CCSD(-)
            3. g_Ai = [E(+) - E(-)] / (2h)

        Args:
            molecule:     Molecule object with geometry in Bohr.
            hf_solver:    Converged HartreeFockSolver (used as template for settings).
            ccsd_solver:  Converged CCSDSolver (used for convergence settings).
            step_size:    Finite difference step size in Bohr (default 0.005).
            verbose:      Print progress information.

        Returns:
            numpy.ndarray of shape (n_atoms, 3) — gradient in Hartree/Bohr.
        """
        atoms = molecule.atoms
        n_atoms = len(atoms)
        gradient = np.zeros((n_atoms, 3))

        # Extract CCSD convergence setting from the solver
        ccsd_convergence = ccsd_solver.convergence
        ccsd_max_iter = ccsd_solver.max_iter

        if verbose:
            print("=" * 60)
            print("CCSD Gradient — Numerical (Central Finite Difference)")
            print(f"  Step size: {step_size} Bohr")
            print(f"  Atoms: {n_atoms}")
            print(f"  Single-point calculations: {6 * n_atoms}")
            print("=" * 60)

        # Create a single working copy of the molecule for displacements
        # (avoids expensive deepcopy per displacement)
        mol_work = deepcopy(molecule)

        for atom_idx in range(n_atoms):
            element, original_pos = atoms[atom_idx]
            original_pos = np.array(original_pos)

            for axis in range(3):
                axis_label = ["x", "y", "z"][axis]

                # --- Forward displacement (+h) ---
                pos_plus = original_pos.copy()
                pos_plus[axis] += step_size

                mol_work.atoms[atom_idx] = (element, tuple(pos_plus))

                hf_plus = HartreeFockSolver()
                E_hf_plus, _ = hf_plus.compute_energy(
                    mol_work, max_iter=100, tolerance=1e-10, verbose=False
                )

                ccsd_plus = CCSDSolver(
                    max_iter=ccsd_max_iter, convergence=ccsd_convergence
                )
                E_ccsd_plus, E_corr_plus = ccsd_plus.solve(
                    hf_plus, mol_work, verbose=False
                )

                E_total_plus = E_hf_plus + E_corr_plus

                # --- Backward displacement (-h) ---
                pos_minus = original_pos.copy()
                pos_minus[axis] -= step_size

                mol_work.atoms[atom_idx] = (element, tuple(pos_minus))

                hf_minus = HartreeFockSolver()
                E_hf_minus, _ = hf_minus.compute_energy(
                    mol_work, max_iter=100, tolerance=1e-10, verbose=False
                )

                ccsd_minus = CCSDSolver(
                    max_iter=ccsd_max_iter, convergence=ccsd_convergence
                )
                E_ccsd_minus, E_corr_minus = ccsd_minus.solve(
                    hf_minus, mol_work, verbose=False
                )

                E_total_minus = E_hf_minus + E_corr_minus

                # Restore original position for next axis
                mol_work.atoms[atom_idx] = (element, tuple(original_pos))

                # --- Central difference ---
                grad_component = (E_total_plus - E_total_minus) / (2.0 * step_size)
                gradient[atom_idx, axis] = grad_component

                if verbose:
                    print(
                        f"  Atom {atom_idx} ({element}) d{axis_label}: "
                        f"E(+) = {E_total_plus:.10f}, "
                        f"E(-) = {E_total_minus:.10f}, "
                        f"grad = {grad_component:+.8f}"
                    )

        if verbose:
            print(f"\nCCSD Gradient (Hartree/Bohr):")
            for i, (el, _) in enumerate(atoms):
                gx, gy, gz = gradient[i]
                print(f"  Atom {i} ({el}): [{gx:+.8f}, {gy:+.8f}, {gz:+.8f}]")
            grad_norm = np.sqrt(np.sum(gradient**2))
            max_grad = np.max(np.abs(gradient))
            print(f"  |grad| = {grad_norm:.8f}, max = {max_grad:.8f}")

        return gradient

    def optimize(
        self,
        molecule,
        basis_name="sto-3g",
        max_steps=10,
        tolerance=1e-4,
        ccsd_convergence=1e-8,
        step_size=0.005,
        verbose=False,
    ):
        """
        Optimize molecular geometry at the CCSD level using BFGS.

        At each optimization step:
            1. Compute CCSD energy at current geometry
            2. Compute numerical CCSD gradient
            3. Update inverse Hessian via BFGS formula
            4. Take a step with backtracking Armijo line search

        Args:
            molecule:           Input Molecule (geometry in Bohr).
            basis_name:         Basis set name (default 'sto-3g').
            max_steps:          Maximum optimization iterations.
            tolerance:          Convergence on max gradient component (Ha/Bohr).
            ccsd_convergence:   CCSD amplitude convergence threshold.
            step_size:          Finite difference step size for gradient.
            verbose:            Print detailed progress.

        Returns:
            (optimized_molecule, history)
            where history is a list of dicts:
                {"step": int, "energy": float, "grad_norm": float,
                 "geometry": list of (element, (x, y, z))}
        """
        if verbose:
            print("=" * 60)
            print("CCSD Geometry Optimization — BFGS")
            print(f"  Basis: {basis_name}")
            print(f"  Max steps: {max_steps}")
            print(f"  Convergence: max |grad| < {tolerance:.1e} Ha/Bohr")
            print(f"  CCSD convergence: {ccsd_convergence:.1e}")
            print("=" * 60)

        current_mol = deepcopy(molecule)
        history = []
        n_atoms = len(current_mol.atoms)
        n_coords = 3 * n_atoms

        # --- Initial energy ---
        hf = HartreeFockSolver()
        E_hf, _ = hf.compute_energy(current_mol, verbose=False)
        ccsd = CCSDSolver(convergence=ccsd_convergence)
        E_ccsd, E_corr = ccsd.solve(hf, current_mol, verbose=False)
        energy = E_hf + E_corr

        # --- Initial gradient ---
        grad_matrix = self.compute_gradient(
            current_mol, hf, ccsd, step_size=step_size, verbose=False
        )
        grad = grad_matrix.flatten()

        grad_norm = np.sqrt(np.sum(grad**2))
        max_grad = np.max(np.abs(grad))

        history.append(
            {
                "step": 0,
                "energy": energy,
                "grad_norm": grad_norm,
                "geometry": deepcopy(current_mol.atoms),
            }
        )

        if verbose:
            print(
                f"\n  Step 0: E = {energy:.10f}, |grad| = {grad_norm:.6f}, "
                f"max = {max_grad:.6f}"
            )

        # Initialize inverse Hessian as identity
        H_inv = np.eye(n_coords)

        for step in range(1, max_steps + 1):
            # --- Check convergence ---
            if max_grad < tolerance:
                if verbose:
                    print(f"\n  Converged at step {step - 1}!")
                    print(f"  Final energy: {energy:.10f} Ha")
                    print(f"  Max gradient: {max_grad:.2e} Ha/Bohr")
                return current_mol, history

            # --- BFGS search direction ---
            p = -H_inv @ grad

            # Ensure descent direction
            directional_deriv = np.dot(grad, p)
            if directional_deriv > 0:
                if verbose:
                    print("  [BFGS] Not descent direction — resetting H_inv")
                H_inv = np.eye(n_coords)
                p = -grad
                directional_deriv = np.dot(grad, p)

            # --- Backtracking line search (Armijo) ---
            x_old = self._atoms_to_flat(current_mol.atoms)
            alpha = 1.0
            c1 = 1e-4
            rho = 0.5
            max_ls = 10

            for ls_iter in range(max_ls):
                x_new = x_old + alpha * p
                trial_mol = self._flat_to_molecule(x_new, current_mol)

                hf_trial = HartreeFockSolver()
                E_hf_trial, _ = hf_trial.compute_energy(trial_mol, verbose=False)
                ccsd_trial = CCSDSolver(convergence=ccsd_convergence)
                _, E_corr_trial = ccsd_trial.solve(hf_trial, trial_mol, verbose=False)
                trial_energy = E_hf_trial + E_corr_trial

                if trial_energy <= energy + c1 * alpha * directional_deriv:
                    break
                alpha *= rho
            else:
                # Line search failed — take small steepest descent step
                if verbose:
                    print("  [BFGS] Line search failed — small SD step")
                alpha = 0.01
                x_new = x_old - alpha * grad
                trial_mol = self._flat_to_molecule(x_new, current_mol)

                hf_trial = HartreeFockSolver()
                E_hf_trial, _ = hf_trial.compute_energy(trial_mol, verbose=False)
                ccsd_trial = CCSDSolver(convergence=ccsd_convergence)
                _, E_corr_trial = ccsd_trial.solve(hf_trial, trial_mol, verbose=False)
                trial_energy = E_hf_trial + E_corr_trial

            # --- New gradient at trial geometry ---
            new_grad_matrix = self.compute_gradient(
                trial_mol, hf_trial, ccsd_trial, step_size=step_size, verbose=False
            )
            new_grad = new_grad_matrix.flatten()

            # --- BFGS Hessian update ---
            s = x_new - x_old
            y = new_grad - grad
            sy = np.dot(s, y)

            if sy > 1e-12:
                rho_bfgs = 1.0 / sy
                I = np.eye(n_coords)
                sy_outer = rho_bfgs * np.outer(s, y)
                ys_outer = rho_bfgs * np.outer(y, s)
                ss_outer = rho_bfgs * np.outer(s, s)
                H_inv = (I - sy_outer) @ H_inv @ (I - ys_outer) + ss_outer
            elif verbose:
                print("  [BFGS] Curvature condition not met — skipping H update")

            # --- Update state ---
            current_mol = trial_mol
            energy = trial_energy
            grad = new_grad
            grad_norm = np.sqrt(np.sum(grad**2))
            max_grad = np.max(np.abs(grad))

            history.append(
                {
                    "step": step,
                    "energy": energy,
                    "grad_norm": grad_norm,
                    "geometry": deepcopy(current_mol.atoms),
                }
            )

            if verbose:
                print(
                    f"  Step {step}: E = {energy:.10f}, |grad| = {grad_norm:.6f}, "
                    f"max = {max_grad:.6f}"
                )

        # --- Check final convergence ---
        if max_grad < tolerance:
            if verbose:
                print(f"\n  Converged at step {max_steps}!")
                print(f"  Final energy: {energy:.10f} Ha")
        elif verbose:
            print(f"\n  WARNING: Not converged in {max_steps} steps")
            print(f"  Final energy: {energy:.10f} Ha")
            print(f"  Max gradient: {max_grad:.2e} Ha/Bohr")

        return current_mol, history

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atoms_to_flat(atoms):
        """Convert atoms list to flat coordinate array."""
        coords = []
        for _, (x, y, z) in atoms:
            coords.extend([x, y, z])
        return np.array(coords, dtype=np.float64)

    @staticmethod
    def _flat_to_molecule(flat_coords, template_mol):
        """Build a new Molecule from flat coordinates, preserving metadata."""
        new_atoms = []
        for i, (el, _) in enumerate(template_mol.atoms):
            x = float(flat_coords[3 * i])
            y = float(flat_coords[3 * i + 1])
            z = float(flat_coords[3 * i + 2])
            new_atoms.append((el, (x, y, z)))
        mol = deepcopy(template_mol)
        mol.atoms = new_atoms
        return mol


# ======================================================================
#  Self-test: H2 gradient at compressed geometry
# ======================================================================


def test_h2_ccsd_gradient():
    """
    Test: H2 at R = 1.2 Bohr (compressed from equilibrium ~1.4 Bohr).

    At this compressed geometry, the total energy gradient should push
    the atoms apart (toward equilibrium). That means:
        - Atom 0 at z=0: gradient in z should be NEGATIVE (force pushes toward -z)
        - Atom 1 at z=1.2: gradient in z should be POSITIVE (force pushes toward +z)

    This is because the energy decreases as the bond stretches toward
    equilibrium, so dE/dz_1 > 0 means moving atom 1 in +z increases
    the coordinate toward equilibrium — wait, gradient points UPHILL.

    Actually: gradient = dE/dR. At compressed geometry, E decreases as R
    increases, so dE/dR < 0. The gradient on atom 1 (at z=R) in the z
    direction is dE/dz_1 = dE/dR < 0. The force = -gradient pushes atom 1
    in +z (away from atom 0), which is correct.

    For atom 0 at z=0: dE/dz_0 = -dE/dR > 0. Force pushes atom 0 in -z.

    So we expect:
        grad[0, 2] > 0  (atom 0, z-direction: positive gradient)
        grad[1, 2] < 0  (atom 1, z-direction: negative gradient)

    By symmetry, x and y gradients should be ~zero.
    """
    print("\n" + "=" * 60)
    print("CCSD Gradient Self-Test: H2 at R = 1.2 Bohr")
    print("=" * 60)

    R = 1.2  # Bohr — compressed from equilibrium
    mol = Molecule(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, R))],
        basis_name="sto-3g",
    )

    # Run HF
    hf = HartreeFockSolver()
    E_hf, _ = hf.compute_energy(mol, verbose=False)
    print(f"  E(HF)  = {E_hf:.10f} Ha")

    # Run CCSD
    ccsd = CCSDSolver(convergence=1e-10)
    E_ccsd, E_corr = ccsd.solve(hf, mol, verbose=False)
    E_total = E_hf + E_corr
    print(f"  E(corr) = {E_corr:.10f} Ha")
    print(f"  E(CCSD) = {E_total:.10f} Ha")

    # Compute gradient
    grad_calc = CCSDGradient()
    gradient = grad_calc.compute_gradient(mol, hf, ccsd, step_size=0.005, verbose=True)

    # --- Assertions ---
    # 1. Shape is correct
    assert gradient.shape == (2, 3), f"Expected shape (2,3), got {gradient.shape}"

    # 2. x and y gradients should be ~zero by symmetry
    assert abs(gradient[0, 0]) < 1e-6, f"Atom 0 x-gradient too large: {gradient[0, 0]}"
    assert abs(gradient[0, 1]) < 1e-6, f"Atom 0 y-gradient too large: {gradient[0, 1]}"
    assert abs(gradient[1, 0]) < 1e-6, f"Atom 1 x-gradient too large: {gradient[1, 0]}"
    assert abs(gradient[1, 1]) < 1e-6, f"Atom 1 y-gradient too large: {gradient[1, 1]}"

    # 3. z-gradient: at compressed geometry, gradient on atom 0 should be > 0
    #    and gradient on atom 1 should be < 0 (energy decreases as bond stretches)
    assert gradient[0, 2] > 0.01, (
        f"Atom 0 z-gradient should be positive (compressed H2): {gradient[0, 2]}"
    )
    assert gradient[1, 2] < -0.01, (
        f"Atom 1 z-gradient should be negative (compressed H2): {gradient[1, 2]}"
    )

    # 4. By translational invariance: sum of gradients should be ~zero
    grad_sum = np.sum(gradient, axis=0)
    assert np.max(np.abs(grad_sum)) < 1e-6, (
        f"Translational invariance violated: sum = {grad_sum}"
    )

    # 5. The z-gradient magnitudes should be equal (by symmetry)
    assert abs(abs(gradient[0, 2]) - abs(gradient[1, 2])) < 1e-6, (
        f"Gradient magnitudes differ: {gradient[0, 2]} vs {gradient[1, 2]}"
    )

    print("\n  All assertions passed!")
    print(f"  H2 CCSD gradient at R=1.2 Bohr:")
    print(f"    dE/dz (atom 0) = {gradient[0, 2]:+.8f} Ha/Bohr")
    print(f"    dE/dz (atom 1) = {gradient[1, 2]:+.8f} Ha/Bohr")
    print(f"    Force pushes atoms apart → toward equilibrium ✓")

    return gradient


if __name__ == "__main__":
    test_h2_ccsd_gradient()
