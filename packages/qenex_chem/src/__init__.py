"""
QENEX Chem (Verifier Subset) — minimal __init__ surface.

The verifier subset uses the lab's "src/ on sys.path with direct imports"
pattern (see conftest.py). Most internal modules use direct imports
(``from molecule import Molecule``), and tests / scripts likewise.

This ``__init__.py`` exists only because the canonical Q-Lang v0.4
``simulate chemistry`` kernel uses a synthetic-module fallback loader
(``packages/qenex_chem/src/__init__.py`` -> ``compute_energy``) when
``qenex_chem`` is not pip-installed. The loader injects ``src/`` into
``sys.path`` before executing this file, so direct imports below resolve
correctly without needing setuptools or relative-import gymnastics.

For the verifier subset, we only re-export the two zero-config entry
points that the v04 chemistry kernel calls into. The rest of the public
chemistry API is reachable via direct imports (``from solver import
HartreeFockSolver``, ``from ccsd import CCSDSolver``, etc.) per the
verifier's ``conftest.py``.

The full lab's ``__init__.py`` exports ~226 symbols across the entire
chemistry, drug-discovery, manufacturing, and regulatory surface; that
file lives in the lab installation and is not appropriate for the
verifier subset.
"""

# Direct imports work because the loader / conftest.py puts src/ on sys.path.
from molecule import Molecule  # noqa: E402,F401  (pyright may flag; runtime is fine)


def compute_energy(
    molecule_or_name,
    method: str = "hf",
    basis: str = "sto-3g",
    charge: int = 0,
    verbose: bool = False,
    frozen_core: bool = False,
):
    """One-call energy computation.

    >>> E, _ = compute_energy("H2")                          # HF/STO-3G
    >>> E, _ = compute_energy("H2O", method="b3lyp", basis="cc-pvdz")
    """
    if isinstance(molecule_or_name, str):
        mol = Molecule.from_name(molecule_or_name, basis_name=basis, charge=charge)
    else:
        mol = molecule_or_name
    return mol.compute(method=method, verbose=verbose, frozen_core=frozen_core)


def compute_energy_geom(
    atoms,
    method: str = "hf",
    basis: str = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    verbose: bool = False,
    frozen_core: bool = False,
):
    """Energy from raw atomic coordinates in Bohr.

    >>> atoms = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.3984))]
    >>> E, _ = compute_energy_geom(atoms)        # ~ -1.1167593074 Ha
    """
    del multiplicity  # accepted-but-unused (forward-compat)
    normalized = []
    for elem, pos in atoms:
        if len(pos) != 3:
            raise ValueError(
                f"compute_energy_geom: atom {elem!r} has {len(pos)} coordinates; expected 3 (x, y, z in Bohr)"
            )
        normalized.append((str(elem), (float(pos[0]), float(pos[1]), float(pos[2]))))
    mol = Molecule(normalized, charge=charge, basis_name=basis)
    return mol.compute(method=method, verbose=verbose, frozen_core=frozen_core)


__all__ = ["compute_energy", "compute_energy_geom", "Molecule"]
