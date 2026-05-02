"""
Molecule Module
Defines chemical structures for quantum simulation.
"""

from typing import List, Tuple


class Molecule:
    """
    Represents a molecular structure.
    """

    def __init__(
        self,
        atoms: List[Tuple[str, Tuple[float, float, float]]],
        charge: int = 0,
        multiplicity: int | None = None,
        basis_name: str = "sto-3g",
    ):
        """
        Initialize molecule with validated atoms, charge, and spin multiplicity.
        """
        # R7-NEW1: Element validation synchronized with solver Z_map (H–Ar, Z=1–18).
        # Previously included Pb(82) and I(53) which pass validation here but produce
        # Z=0 in solver, causing silently wrong nuclear repulsion and electron counts.
        # Only elements with full basis set support (STO-3G through 3rd row) are allowed.
        valid_elements = set(
            [
                "H",
                "He",
                "Li",
                "Be",
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
            ]
        )  # Rows 1-3 of periodic table (Z=1-18)

        atomic_numbers = {
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

        if not atoms:
            raise ValueError("Molecule Error: atom list is empty.")

        # Validate coordinates are finite
        import math

        for i, (element, coords) in enumerate(atoms):
            if len(coords) != 3:
                raise ValueError(
                    f"Atom {i} ({element}): expected 3 coordinates, got {len(coords)}"
                )
            for j, c in enumerate(coords):
                if not isinstance(c, (int, float)):
                    raise ValueError(
                        f"Atom {i} ({element}): coordinate {j} is not a number: {c}"
                    )
                if math.isnan(c) or math.isinf(c):
                    raise ValueError(
                        f"Atom {i} ({element}): coordinate {j} is {'NaN' if math.isnan(c) else 'Inf'}"
                    )

        for element, _ in atoms:
            if element not in valid_elements:
                raise ValueError(f"Alchemy Error: Unknown element '{element}'.")

        # [FIX] Auto-detect multiplicity for spin parity
        # Total electrons = Sum(Z) - charge
        total_protons = sum(atomic_numbers[element] for element, _ in atoms)
        total_electrons = total_protons - charge

        # If multiplicity not provided, auto-detect based on electron count
        if multiplicity is None:
            if total_electrons == 0:
                multiplicity = 1  # Bare nucleus: vacuous singlet (no electrons to pair)
            elif total_electrons % 2 == 0:
                multiplicity = 1  # Singlet for even electrons (default lowest spin)
            else:
                multiplicity = 2  # Doublet for odd electrons (default lowest spin)

        # [SECURITY PATCH] Spin Multiplicity Check (2S+1 rule)
        # If Ne is Even, Multiplicity must be Odd (1, 3, 5...)
        # If Ne is Odd, Multiplicity must be Even (2, 4, 6...)
        # Skip check for bare nuclei (zero electrons — vacuously consistent)
        if total_electrons > 0:
            if total_electrons % 2 == 0:
                if multiplicity % 2 == 0:
                    raise ValueError(
                        f"Spin Parity Error: Even electrons ({total_electrons}) require Odd multiplicity (Got {multiplicity})."
                    )
            else:
                if multiplicity % 2 != 0:
                    raise ValueError(
                        f"Spin Parity Error: Odd electrons ({total_electrons}) require Even multiplicity (Got {multiplicity})."
                    )

        # Validate electron count — negative is always invalid; zero = bare nucleus (allowed)
        if total_electrons < 0:
            raise ValueError(
                f"Invalid electron count: {total_electrons}. "
                f"Charge {charge} is too large for this molecule "
                f"(total nuclear charge = {total_protons}). "
                f"Maximum charge for this molecule is +{total_protons}."
            )

        # Validate multiplicity vs electron count
        max_mult = total_electrons + 1
        if multiplicity > max_mult:
            raise ValueError(
                f"Invalid multiplicity {multiplicity} for {total_electrons} electrons. "
                f"Maximum multiplicity is {max_mult} "
                f"(requires {multiplicity - 1} unpaired electrons, but only {total_electrons} electrons available)."
            )

        # Validate that n_alpha and n_beta are both non-negative
        n_unpaired = multiplicity - 1
        n_paired_electrons = total_electrons - n_unpaired
        if n_paired_electrons < 0 or n_paired_electrons % 2 != 0:
            raise ValueError(
                f"Inconsistent electron count ({total_electrons}) and multiplicity ({multiplicity}). "
                f"Cannot assign {n_unpaired} unpaired electrons from {total_electrons} total."
            )
        n_alpha = (total_electrons + n_unpaired) // 2
        n_beta = (total_electrons - n_unpaired) // 2
        if n_beta < 0:
            raise ValueError(
                f"Invalid: multiplicity {multiplicity} requires {n_alpha} alpha and {n_beta} beta electrons, "
                f"but beta count cannot be negative."
            )

        # Ensure coordinates are floats (required by Rust FFI)
        self.atoms = [(el, (float(x), float(y), float(z))) for el, (x, y, z) in atoms]
        self.charge = charge
        self.multiplicity = multiplicity
        self.basis_name = basis_name

    def __repr__(self):
        """Return string representation with molecular formula and charge."""
        formula = "".join([a[0] for a in self.atoms])
        return f"Molecule({formula}, charge={self.charge})"

    # ------------------------------------------------------------------
    # Zero-config factory helpers
    # ------------------------------------------------------------------

    # Built-in geometries (Bohr) at experimental equilibrium.
    #
    # IMPORTANT: all coordinates are in atomic units (Bohr).  The
    # solver reads them as Bohr (see ``compute_hf_with_libcint`` which
    # passes ``unit="bohr"`` to PySCF).  Storing a molecule in
    # Angstroms here silently compresses the geometry by a factor of
    # 1.8897, producing catastrophically wrong energies.
    #
    # Reference values (from NIST CCCBDB; https://cccbdb.nist.gov):
    #   H2:  r(H-H) = 0.7414 Å = 1.3984 Bohr
    #   H2O: r(O-H) = 0.9572 Å = 1.8091 Bohr, H-O-H = 104.52°
    #   NH3: r(N-H) = 1.0124 Å = 1.9131 Bohr, H-N-H = 106.67°
    #   CH4: r(C-H) = 1.0870 Å = 2.0541 Bohr, tetrahedral
    #
    # When adding a new molecule here, cite the source in a comment
    # and double-check the numerical value(s) are in Bohr, not Å.
    _BUILTINS: dict = {
        "He": [("He", (0.0, 0.0, 0.0))],
        # H2: r(H-H) = 1.3984 Bohr
        "H2": [
            ("H", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.3984)),
        ],
        # H2O: R(O-H) = 1.80911 Bohr, theta(H-O-H) = 104.52°
        # H atoms placed symmetrically in the y-z plane so the
        # C2v axis is z.  Computed from experimental geometry.
        "H2O": [
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 1.430638, 1.107319)),
            ("H", (0.0, -1.430638, 1.107319)),
        ],
        # NH3: R(N-H) = 1.9132 Bohr (= 1.0124 Å), H-N-H = 106.67°
        # (NIST CCCBDB experimental geometry).  The previous entry had
        # R = 1.9005 Bohr which corresponds to R(N-H) ≈ 1.006 Å —
        # about 0.007 Å short of experiment.  Fixed in the scientific-
        # reference audit (2026-04-21).
        "NH3": [
            ("N", (0.0, 0.0, 0.0)),
            ("H", (0.000000, 1.772029, -0.721172)),
            ("H", (1.534622, -0.886014, -0.721172)),
            ("H", (-1.534622, -0.886014, -0.721172)),
        ],
        # CH4: tetrahedral, R(C-H) = 2.0541 Bohr.  Previous entry is
        # already in Bohr.
        "CH4": [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (1.1860, 1.1860, 1.1860)),
            ("H", (-1.1860, -1.1860, 1.1860)),
            ("H", (-1.1860, 1.1860, -1.1860)),
            ("H", (1.1860, -1.1860, -1.1860)),
        ],
    }

    @classmethod
    def from_name(
        cls,
        name: str,
        basis_name: str = "sto-3g",
        charge: int = 0,
        multiplicity: int | None = None,
    ) -> "Molecule":
        """Create a Molecule from a built-in name (He, H2, H2O, NH3, CH4).

        Example::

            mol = Molecule.from_name("H2O", basis_name="cc-pvdz")
        """
        key = name.strip()
        if key not in cls._BUILTINS:
            raise ValueError(
                f"Unknown molecule name {key!r}. "
                f"Built-ins: {list(cls._BUILTINS.keys())}. "
                f"Pass atoms= list directly for custom geometries."
            )
        return cls(
            cls._BUILTINS[key],
            charge=charge,
            multiplicity=multiplicity,
            basis_name=basis_name,
        )

    def compute(
        self,
        method: str = "hf",
        verbose: bool = False,
        frozen_core: bool = False,
    ) -> tuple:
        """Zero-config energy calculation.  Returns (E_total, E_electronic).

        Supported methods: ``"hf"`` (default), ``"dft"``, ``"lda"``,
        ``"b3lyp"``, ``"mp2"``, ``"ccsd"``.

        ``frozen_core`` freezes the inner 1s core orbitals for post-HF
        correlation methods (MP2, CCSD).  Matches the convention in
        standard reference tables (e.g. Helgaker et al. 2000 Tables
        15.1-15.3) and the ``*`` frozen-core suffix in common basis
        sets like ``cc-pVDZ``.  Defaults to False (all-electron);
        set True to match textbook reference values.

        Example::

            mol = Molecule.from_name("H2")
            E_total, E_elec = mol.compute()                      # HF/STO-3G
            E_total, E_elec = mol.compute("b3lyp")               # DFT B3LYP
            E_total, E_elec = mol.compute("mp2", frozen_core=True)
        """
        m = method.lower()
        if m in ("hf", "rhf"):
            from solver import HartreeFockSolver

            return HartreeFockSolver().compute_energy(self, verbose=verbose)
        if m in ("dft", "lda"):
            from dft import DFTSolver

            return DFTSolver(
                self, functional="LDA", basis=self.basis_name
            ).compute_energy()
        if m == "b3lyp":
            from dft import DFTSolver

            return DFTSolver(
                self, functional="B3LYP", basis=self.basis_name
            ).compute_energy()
        if m in ("mp2", "ccsd"):
            # MP2 and CCSD share the CCSDSolver path: CCSD's iterative
            # amplitudes include MP2 as the first-iteration estimate.
            # For MP2 we stop after the MP2 energy is computed; for
            # CCSD we iterate to convergence.
            from solver import HartreeFockSolver
            from ccsd import CCSDSolver

            hf = HartreeFockSolver()
            hf.compute_energy(self, verbose=verbose)

            if m == "mp2":
                # First iteration of CCSD gives the MP2 correlation
                # energy; run with max_iter=1 to stop after MP2.
                solver_mp2 = CCSDSolver(
                    max_iter=1,
                    convergence=1e-12,
                    frozen_core=frozen_core,
                )
                E_total, E_corr = solver_mp2.solve(hf, self, verbose=verbose)
                return E_total, E_total - E_corr
            # CCSD: full iteration to convergence.
            solver = CCSDSolver(frozen_core=frozen_core)
            E_total, E_corr = solver.solve(hf, self, verbose=verbose)
            return E_total, E_total - E_corr
        raise ValueError(
            f"Unknown method {method!r}. Supported: hf, dft, lda, b3lyp, mp2, ccsd."
        )
