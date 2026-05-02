"""
Q-Lang v0.4 — simulate-block dispatcher (SPEC §9).

``simulate DOMAIN { kwargs }`` expressions are routed here.  The
dispatcher:

1. Validates the domain is registered.
2. Unwraps Q-Lang ``Value``s into Python-native types the kernel
   expects (``str``, ``float``, etc.) while carrying the node_ids
   through so the result node links back to them.
3. Calls the registered kernel callable.
4. Wraps the result back into a ``Quantity`` (or ``Host``) with a
   ``kernel_node`` in the trace, producer tagged with the kernel
   version.

The pre-existing ``simulation_handlers.py`` has 30+ ``_sim_*``
handlers using a string-parsed ``parts`` calling convention that is
(a) not kwarg-shaped, (b) swallows kernel errors in bare ``except``.
We do NOT route through that class for v0.4.  Instead each v0.4 domain
has a dedicated wrapper that calls the underlying QENEX kernel
directly (clean kwarg interface, honest error propagation).

v0.4 registered domains:
  * ``chemistry``       — named built-ins via ``compute_energy(name, ...)``
  * ``chemistry_geom``  — arbitrary geometries via ``compute_energy_geom``
                          (Path A: verifies discovery-engine output)
  * ``md``              — minimal molecular dynamics

Adding another domain is one ``register_domain(...)`` call.  Deliberate
minimalism: each domain's wrapper must be hand-crafted to get the call
surface, unit handling, and error semantics right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from errors_v04 import (  # type: ignore[import-not-found]
    ConservationViolation,
    DimensionMismatchError,
    InvariantViolation,
    KernelError,
    QLangSyntaxError,
)
from provenance_v04 import Trace, kernel_node  # type: ignore[import-not-found]
from units_v04 import Dim  # type: ignore[import-not-found]
from values_v04 import Host, Quantity, String  # type: ignore[import-not-found]


# ─────────────────────────────────────────────────────────────────────
# Domain registry
# ─────────────────────────────────────────────────────────────────────


@dataclass
class DomainHandler:
    """One registered simulate-domain wrapper.

    ``call(kwargs, input_ids, trace, line, col)`` -> Value
    """

    name: str
    call: Callable[..., Any]
    producer: str


_REGISTRY: Dict[str, DomainHandler] = {}


def register_domain(name: str, call: Callable[..., Any], producer: str) -> None:
    """Register a simulate DOMAIN {...} handler."""
    _REGISTRY[name] = DomainHandler(name=name, call=call, producer=producer)


def dispatch(
    *,
    domain: str,
    kwargs: Dict[str, Any],
    input_ids: Tuple[str, ...],
    line: Optional[int],
    col: Optional[int],
    trace: Trace,
) -> Any:
    """Entry point called from ``Evaluator._eval_simulate``."""
    if domain not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise QLangSyntaxError(
            f"unknown simulate domain {domain!r}; known: {known}",
            line=line,
            col=col,
        )
    handler = _REGISTRY[domain]
    try:
        return handler.call(
            kwargs=kwargs,
            input_ids=input_ids,
            trace=trace,
            line=line,
            col=col,
            producer=handler.producer,
        )
    except (
        DimensionMismatchError,
        QLangSyntaxError,
        KernelError,
        ConservationViolation,
        InvariantViolation,
    ):
        # Typed Q-Lang errors propagate unchanged.
        raise
    except Exception as e:
        raise KernelError(
            f"simulate {domain}",
            e,
            line=line,
            col=col,
        )


# ─────────────────────────────────────────────────────────────────────
# Helpers for domain wrappers
# ─────────────────────────────────────────────────────────────────────


def _get_string(
    name: str, kwargs: Dict[str, Any], *, line: Optional[int], col: Optional[int]
) -> str:
    if name not in kwargs:
        raise QLangSyntaxError(
            f"simulate argument {name!r} is required",
            line=line,
            col=col,
        )
    v = kwargs[name]
    if isinstance(v, String):
        return v.value
    raise DimensionMismatchError(
        f"simulate argument {name!r} must be a string, got {type(v).__name__}",
        line=line,
        col=col,
    )


def _get_string_optional(name: str, kwargs: Dict[str, Any]) -> Optional[str]:
    if name not in kwargs:
        return None
    v = kwargs[name]
    if isinstance(v, String):
        return v.value
    return None


def _load_qenex_chem_function(name: str):
    """Load a named function from the QENEX chemistry package.

    Tries two layouts in order:
      1. ``import qenex_chem; getattr(qenex_chem, name)`` (when
         the package has been installed / otherwise on sys.path).
      2. ``importlib`` load of
         ``packages/qenex_chem/src/__init__.py`` as a synthetic
         module (workspace-in-tree development mode).

    Returns the callable, or None if neither path works.
    """
    import importlib
    import importlib.util
    import os
    import sys

    try:
        mod = importlib.import_module("qenex_chem")
        fn = getattr(mod, name, None)
        if fn is not None:
            return fn
    except ImportError:
        pass

    # In-tree fallback: load the __init__.py as a synthetic module.
    here = os.path.dirname(os.path.abspath(__file__))
    init_path = os.path.normpath(
        os.path.join(
            here,
            "..",
            "..",
            "..",
            "qenex_chem",
            "src",
            "__init__.py",
        )
    )
    if not os.path.isfile(init_path):
        return None

    # Ensure the src dir is on sys.path so that __init__.py's own
    # ``from solver import ...`` statements resolve.
    src_dir = os.path.dirname(init_path)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        # Reuse a shared synthetic module across both lookups so the
        # ``compute_energy`` and ``compute_energy_geom`` paths converge
        # on the same import — avoids loading the chemistry stack twice.
        bridge_name = "_qenex_chem_v4_bridge"
        if bridge_name in sys.modules:
            return getattr(sys.modules[bridge_name], name, None)
        spec = importlib.util.spec_from_file_location(bridge_name, init_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[bridge_name] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, name, None)
    except Exception:
        return None


def _load_compute_energy():
    """Load ``compute_energy`` from the QENEX chemistry package.

    Wrapper kept for backward compatibility — see
    :func:`_load_qenex_chem_function` for the underlying loader.
    """
    return _load_qenex_chem_function("compute_energy")


def _load_compute_energy_geom():
    """Load ``compute_energy_geom`` from the QENEX chemistry package.

    Added 2026-05-01 for the v0.4 ``simulate chemistry_geom`` block.
    See :func:`_call_chemistry_geom`.
    """
    return _load_qenex_chem_function("compute_energy_geom")


# ─────────────────────────────────────────────────────────────────────
# Chemistry domain wrapper
# ─────────────────────────────────────────────────────────────────────


def _call_chemistry(
    *,
    kwargs: Dict[str, Any],
    input_ids: Tuple[str, ...],
    trace: Trace,
    line: Optional[int],
    col: Optional[int],
    producer: str,
) -> Any:
    """``simulate chemistry { molecule, method, basis }`` —
    routes to ``qenex_chem.compute_energy(name, method, basis)``.

    Returns a ``Quantity`` in ``[Hartree]`` with a ``simulate.chemistry``
    node in the trace.
    """

    molecule = _get_string("molecule", kwargs, line=line, col=col)
    method = _get_string_optional("method", kwargs) or "hf"
    basis = _get_string_optional("basis", kwargs) or "sto-3g"

    # Import here so ImportError is handled as a KernelError, not a
    # module-load-time crash.  The existing QENEX layout has the
    # chemistry modules at ``packages/qenex_chem/src/`` and clients
    # typically do ``from solver import ...`` with sys.path tweaks
    # rather than importing a top-level ``qenex_chem`` package.
    compute_energy = _load_compute_energy()
    if compute_energy is None:
        raise KernelError(
            "qenex_chem.compute_energy",
            ImportError("qenex_chem.compute_energy is not importable"),
            line=line,
            col=col,
        )

    try:
        result = compute_energy(molecule, method=method, basis=basis)
    except Exception as e:
        raise KernelError(
            "qenex_chem.compute_energy",
            e,
            line=line,
            col=col,
        )

    try:
        result = compute_energy(molecule, method=method, basis=basis)
    except Exception as e:
        raise KernelError(
            "qenex_chem.compute_energy",
            e,
            line=line,
            col=col,
        )

    # qenex_chem returns (E_total, meta_dict) or similar — normalise.
    if isinstance(result, tuple) and len(result) >= 1:
        e_raw: Any = result[0]
    else:
        e_raw = result

    if not isinstance(e_raw, (int, float)):
        raise KernelError(
            "qenex_chem.compute_energy",
            TypeError(f"expected numeric energy, got {type(e_raw).__name__}"),
            line=line,
            col=col,
        )
    e_float = float(e_raw)

    # Hartree energy dimension: ML²T⁻²
    hartree_dim = Dim(mass=1, length=2, time=-2)
    # Internal SI magnitude (Hartree -> J)
    e_si = e_float * 4.3597447222071e-18

    meta = {
        "domain": "chemistry",
        "molecule": molecule,
        "method": method,
        "basis": basis,
    }
    node = kernel_node(
        op="simulate.chemistry",
        input_ids=input_ids,
        magnitude=e_si,
        unit="J",
        uncertainty=None,
        producer=producer,
        meta=meta,
    )
    trace.push(node)
    return Quantity(
        magnitude=e_si,
        dim=hartree_dim,
        uncertainty=None,
        node_id=node.node_id,
        display_unit="J",
    )


# ─────────────────────────────────────────────────────────────────────
# Chemistry geometry domain wrapper (Path A — verify discoveries)
# ─────────────────────────────────────────────────────────────────────


def _call_chemistry_geom(
    *,
    kwargs: Dict[str, Any],
    input_ids: Tuple[str, ...],
    trace: Trace,
    line: Optional[int],
    col: Optional[int],
    producer: str,
) -> Any:
    """``simulate chemistry_geom { atoms_file, method, basis, charge, multiplicity }``
    — routes to ``qenex_chem.compute_energy_geom(atoms, method, basis, ...)``.

    Loads atomic geometry from a JSON file (since v0.4 grammar lacks
    list literals), then computes the electronic energy with the
    specified method/basis. Returns a ``Quantity`` in ``[J]`` (with
    Hartree dimension MLT⁻²) and adds a ``simulate.chemistry_geom``
    node to the trace.

    Reproducibility use case: a discovery's ``atoms.json`` geometry
    sidecar can be paired with a ``.qlang`` verification program that
    reads it via this block; ``qlang replay`` of the resulting trace
    then proves the discovery's electronic-structure result is
    reproducible from the raw coordinates.

    Required arguments:
        atoms_file : STRING — absolute or relative path to a JSON file
                              containing ``[[elem, [x, y, z]], ...]`` in Bohr.
                              Relative paths are interpreted from the .ql
                              file's directory (i.e. the same place an
                              ``atoms.json`` sidecar would live).

    Optional arguments:
        method       : STRING (default ``"hf"``)
        basis        : STRING (default ``"sto-3g"``)
        charge       : INT-coercible STRING (default ``"0"``)
        multiplicity : INT-coercible STRING (default ``"1"``)

    The numeric kwargs are accepted as strings to match v0.4's existing
    simulate-block convention (the dispatcher's helpers operate on String
    values; integer-via-string is the lowest-friction path until v0.4's
    grammar supports integer literals in simulate blocks).
    """
    import json as _json
    import os as _os

    atoms_file_arg = _get_string("atoms_file", kwargs, line=line, col=col)
    method = _get_string_optional("method", kwargs) or "hf"
    basis = _get_string_optional("basis", kwargs) or "sto-3g"
    charge_str = _get_string_optional("charge", kwargs) or "0"
    multiplicity_str = _get_string_optional("multiplicity", kwargs) or "1"

    try:
        charge = int(charge_str)
    except ValueError as e:
        raise KernelError(
            "simulate.chemistry_geom",
            ValueError(f"charge must be an integer, got {charge_str!r}"),
            line=line,
            col=col,
        ) from e

    try:
        multiplicity = int(multiplicity_str)
    except ValueError as e:
        raise KernelError(
            "simulate.chemistry_geom",
            ValueError(f"multiplicity must be an integer, got {multiplicity_str!r}"),
            line=line,
            col=col,
        ) from e

    # Resolve atoms_file: if relative, prefer cwd then trace's source-file dir.
    # We don't currently thread the .ql file path through to here, so cwd is the
    # most reliable anchor — call sites typically `cd` to the discovery's dir
    # before running ``qlang run``.
    if not _os.path.isabs(atoms_file_arg):
        candidates = [
            atoms_file_arg,  # relative to cwd
            _os.path.abspath(atoms_file_arg),
        ]
        atoms_file = next((p for p in candidates if _os.path.isfile(p)), atoms_file_arg)
    else:
        atoms_file = atoms_file_arg

    if not _os.path.isfile(atoms_file):
        raise KernelError(
            "simulate.chemistry_geom",
            FileNotFoundError(f"atoms_file not found: {atoms_file_arg!r}"),
            line=line,
            col=col,
        )

    try:
        with open(atoms_file, "r") as f:
            atoms_raw = _json.load(f)
    except _json.JSONDecodeError as e:
        raise KernelError(
            "simulate.chemistry_geom",
            ValueError(f"atoms_file {atoms_file!r} is not valid JSON: {e}"),
            line=line,
            col=col,
        ) from e
    except OSError as e:
        raise KernelError(
            "simulate.chemistry_geom",
            e,
            line=line,
            col=col,
        ) from e

    if not isinstance(atoms_raw, list) or not atoms_raw:
        raise KernelError(
            "simulate.chemistry_geom",
            ValueError(
                f"atoms_file {atoms_file!r} must contain a non-empty list of [elem, [x, y, z]] entries"
            ),
            line=line,
            col=col,
        )

    compute_energy_geom = _load_compute_energy_geom()
    if compute_energy_geom is None:
        raise KernelError(
            "qenex_chem.compute_energy_geom",
            ImportError(
                "qenex_chem.compute_energy_geom is not importable — "
                "this requires Step 4 of the Path A productionization"
            ),
            line=line,
            col=col,
        )

    try:
        result = compute_energy_geom(
            atoms_raw,
            method=method,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )
    except Exception as e:
        raise KernelError(
            "qenex_chem.compute_energy_geom",
            e,
            line=line,
            col=col,
        ) from e

    # qenex_chem returns (E_total, E_electronic) — take the total.
    if isinstance(result, tuple) and len(result) >= 1:
        e_raw: Any = result[0]
    else:
        e_raw = result

    if not isinstance(e_raw, (int, float)):
        raise KernelError(
            "qenex_chem.compute_energy_geom",
            TypeError(f"expected numeric energy, got {type(e_raw).__name__}"),
            line=line,
            col=col,
        )
    e_float = float(e_raw)

    hartree_dim = Dim(mass=1, length=2, time=-2)
    e_si = e_float * 4.3597447222071e-18  # Hartree -> J

    meta = {
        "domain": "chemistry_geom",
        "atoms_file": atoms_file_arg,  # original (relative or absolute) for portability
        "n_atoms": len(atoms_raw),
        "method": method,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
    }
    node = kernel_node(
        op="simulate.chemistry_geom",
        input_ids=input_ids,
        magnitude=e_si,
        unit="J",
        uncertainty=None,
        producer=producer,
        meta=meta,
    )
    trace.push(node)
    return Quantity(
        magnitude=e_si,
        dim=hartree_dim,
        uncertainty=None,
        node_id=node.node_id,
        display_unit="J",
    )


# ─────────────────────────────────────────────────────────────────────
# Auto-register the v0.4 default domains
# ─────────────────────────────────────────────────────────────────────


def _qenex_chem_producer() -> str:
    """Build a 'producer' string that identifies the qenex_chem
    version.  Used for provenance + replay-verifier matching."""
    try:
        import qenex_chem  # type: ignore[import-not-found]

        ver = getattr(qenex_chem, "__version__", None)
        if ver:
            return f"qenex_chem@{ver}"
    except Exception:
        pass

    # Fall back to the git SHA of qenex-lab at import time.
    try:
        import subprocess

        sha = (
            subprocess.check_output(
                ["git", "-C", _git_repo_root(), "rev-parse", "--short", "HEAD"],
                timeout=2.0,
            )
            .decode()
            .strip()
        )
        return f"qenex_chem@{sha}"
    except Exception:
        return "qenex_chem@unknown"


def _git_repo_root() -> str:
    import os

    here = os.path.abspath(os.path.dirname(__file__))
    # up to workspace root
    for _ in range(10):
        if os.path.isdir(os.path.join(here, ".git")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return os.path.abspath(os.path.dirname(__file__))


def _register_defaults() -> None:
    register_domain(
        "chemistry",
        _call_chemistry,
        producer=_qenex_chem_producer(),
    )
    register_domain(
        "chemistry_geom",
        _call_chemistry_geom,
        producer=_qenex_chem_producer(),
    )
    register_domain(
        "md",
        _call_md,
        producer="qenex_qlang_v0.4.md",
    )


# ─────────────────────────────────────────────────────────────────────
# Molecular dynamics domain (minimal v0.4 implementation)
# ─────────────────────────────────────────────────────────────────────

# Conserved-quantity names v0.4 recognises in ``conserve:`` arguments.
# Matches conservation_analyzer.ConservationLaw enum values.
_KNOWN_CONSERVED = frozenset(
    {
        "total_energy",
        "linear_momentum",
        "angular_momentum",
        "charge",
        "entropy",
        "mass",
    }
)


def _call_md(
    *,
    kwargs: Dict[str, Any],
    input_ids: Tuple[str, ...],
    trace: Trace,
    line: Optional[int],
    col: Optional[int],
    producer: str,
) -> Any:
    """``simulate md { system, method, n_steps, conserve:, tolerance: }``

    v0.4 implementation scope:

    * Runs a NVE velocity-Verlet integration of an Lennard-Jones
      Ar-dimer system, hand-rolled here so energy conservation is
      structural (no thermostat).  This is the ONLY ``system`` v0.4
      recognises; all other systems raise ``QLangSyntaxError`` with
      a message naming the valid set.
    * If ``conserve:`` is present, the trajectory is streamed through
      a ``TrajectoryGuard`` that aborts with
      ``ConservationViolation`` when drift exceeds ``tolerance``.
    * Returns a ``Host`` wrapping a lightweight trajectory object
      (``.frames``, ``.energy_drift``, ``.n_steps``) for programs
      that want to inspect the run afterward.
    * The trace records one ``simulate.md`` node with the initial
      energy, final energy, and drift — enough for ``qlang replay``
      to re-verify bitwise on a fixed-seed run.

    Larger / more diverse MD workloads are v0.5+.  v0.4's job here
    is to prove that conservation enforcement is a LANGUAGE-level
    guarantee in action, not a sentiment.
    """

    from errors_v04 import (  # type: ignore[import-not-found]
        ConservationViolation,
        DimensionMismatchError,
    )
    # KernelError already imported at module top
    # Host / Quantity / String already imported at module top

    system_name = _get_string("system", kwargs, line=line, col=col)
    method = _get_string_optional("method", kwargs) or "lennard-jones"

    # ``n_steps`` is a Quantity now (all numeric literals are Quantities
    # in v0.4).  Extract the integer magnitude and check dimensionless.
    if "n_steps" not in kwargs:
        raise QLangSyntaxError(
            "simulate md: required kwarg 'n_steps' missing",
            line=line,
            col=col,
        )
    n_steps_v = kwargs["n_steps"]
    if not isinstance(n_steps_v, Quantity):
        raise DimensionMismatchError(
            f"simulate md: 'n_steps' must be a dimensionless integer, "
            f"got {type(n_steps_v).__name__}",
            line=line,
            col=col,
        )
    if not n_steps_v.dim.is_dimensionless():
        raise DimensionMismatchError(
            f"simulate md: 'n_steps' must be dimensionless, got [{n_steps_v.dim}]",
            have=str(n_steps_v.dim),
            need="1",
            line=line,
            col=col,
        )
    n_steps = int(round(n_steps_v.magnitude))
    if n_steps <= 0:
        raise DimensionMismatchError(
            f"simulate md: 'n_steps' must be > 0, got {n_steps}",
            line=line,
            col=col,
        )

    # ``conserve:`` arrives as a String after the evaluator's
    # symbolic-tag pass (see evaluator_v04._SYMBOLIC_TAG_KWARGS).
    conserve_name: Optional[str] = None
    cv = kwargs.get("conserve")
    if isinstance(cv, String):
        conserve_name = cv.value  # type: ignore[attr-defined]

    # ``tolerance:`` is a Quantity.  Interpret its magnitude in the
    # same "internal" unit as the MD integrator produces
    # (Lennard-Jones reduced units for v0.4 — dimensionless internally,
    # but the tolerance Quantity's dim must match the monitored quantity).
    tolerance_v: Optional[Quantity] = None
    tv = kwargs.get("tolerance")
    if tv is not None:
        if not isinstance(tv, Quantity):
            raise DimensionMismatchError(
                "simulate md: 'tolerance' must be a Quantity",
                line=line,
                col=col,
            )
        tolerance_v = tv

    # v0.4 systems whitelist.
    if system_name not in ("Ar-dimer",):
        raise QLangSyntaxError(
            f"simulate md: unknown system {system_name!r}; v0.4 accepts "
            f"only 'Ar-dimer'. Larger systems land in v0.5+.",
            line=line,
            col=col,
        )
    if method not in ("lennard-jones",):
        raise QLangSyntaxError(
            f"simulate md: unknown method {method!r}; v0.4 accepts only "
            f"'lennard-jones'.",
            line=line,
            col=col,
        )

    # Run the integrator.
    try:
        result = _run_lj_ar_dimer_nve(
            n_steps=n_steps,
            conserve=conserve_name,
            tolerance=(tolerance_v.magnitude if tolerance_v is not None else None),
        )
    except ConservationViolation:
        raise
    except Exception as e:
        raise KernelError(
            "simulate md (Ar-dimer)",
            e,
            line=line,
            col=col,
        )

    # Record one MD node in the trace.
    meta = {
        "domain": "md",
        "system": system_name,
        "method": method,
        "n_steps": n_steps,
        "conserve": conserve_name or "",
        "tolerance": (tolerance_v.magnitude if tolerance_v is not None else None),
        "energy_drift": result["energy_drift"],
        "E_initial": result["E_initial"],
        "E_final": result["E_final"],
    }
    node = kernel_node(
        op="simulate.md",
        input_ids=input_ids,
        magnitude=result["energy_drift"],
        unit="Hartree",
        uncertainty=None,
        producer=producer,
        meta=meta,
    )
    trace.push(node)

    # Wrap the trajectory in a Host so the user can inspect it
    # programmatically (e.g. invariant: traj.energy_drift < ...).
    from units_v04 import Dim  # type: ignore[import-not-found]

    return Host(payload=_LJTrajectory(**result), dim=Dim(), node_id=node.node_id)


# ─────────────────────────────────────────────────────────────────────
# Minimal LJ Ar-dimer NVE integrator (hand-rolled, v0.4-scope)
# ─────────────────────────────────────────────────────────────────────
#
# Reduced units: σ = ε = m = 1.  Positions in σ, velocities in σ/τ_LJ.
# Time step dt = 0.005 τ_LJ is well within the stability envelope for
# velocity-Verlet on LJ-12-6.  A 1000-step run conserves total energy
# to ~10⁻⁷ in reduced units for an equilibrated dimer.


class _LJTrajectory:
    """Lightweight trajectory container for v0.4 MD runs."""

    __slots__ = ("frames", "energy_drift", "E_initial", "E_final", "n_steps", "dt")

    def __init__(self, frames, energy_drift, E_initial, E_final, n_steps, dt):
        self.frames = frames
        self.energy_drift = energy_drift
        self.E_initial = E_initial
        self.E_final = E_final
        self.n_steps = n_steps
        self.dt = dt


class _LJFrame:
    """One recorded step of the LJ Ar-dimer NVE integration.

    Exposes the attributes TrajectoryGuard samples: E_total,
    positions, velocities, masses.  No charges (LJ system is neutral).
    """

    __slots__ = (
        "step",
        "time",
        "positions",
        "velocities",
        "masses",
        "E_kinetic",
        "E_potential",
        "E_total",
    )

    def __init__(
        self, step, time, positions, velocities, masses, E_kinetic, E_potential
    ):
        self.step = step
        self.time = time
        self.positions = positions
        self.velocities = velocities
        self.masses = masses
        self.E_kinetic = float(E_kinetic)
        self.E_potential = float(E_potential)
        self.E_total = float(E_kinetic + E_potential)


def _lj_force_and_energy(r):
    """For a 2-atom system with positions r = [[x0,y0,z0],[x1,y1,z1]]
    in reduced LJ units, return (force[2][3], E_pot)."""
    import math

    dx = r[1][0] - r[0][0]
    dy = r[1][1] - r[0][1]
    dz = r[1][2] - r[0][2]
    r2 = dx * dx + dy * dy + dz * dz
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    E_pot = 4.0 * (inv_r12 - inv_r6)
    # F = -dU/dr on atom 0; atom 1 gets -F.
    # dU/dr = 4 * (-12/r^13 + 6/r^7) = -24/r^13 (2/r^6 - 1)
    # Force magnitude along separation:
    fmag_over_r = 24.0 * inv_r2 * inv_r6 * (2.0 * inv_r6 - 1.0)
    fx = -fmag_over_r * dx
    fy = -fmag_over_r * dy
    fz = -fmag_over_r * dz
    # Force on atom 0: points from 0 towards 1 → +(dx,dy,dz) when
    # repulsive.  The sign above follows from F_on_0 = -dU/dr_0 =
    # +dU/dr_1 convention.
    F = [[fx, fy, fz], [-fx, -fy, -fz]]
    return F, E_pot


def _run_lj_ar_dimer_nve(
    *, n_steps: int, conserve: Optional[str], tolerance: Optional[float]
) -> Dict[str, Any]:
    """Velocity-Verlet NVE on a 2-particle LJ system (Ar dimer).

    Returns a dict with keys ``frames, energy_drift, E_initial,
    E_final, n_steps, dt``.  Raises ``ConservationViolation`` (from
    TrajectoryGuard) if drift exceeds tolerance partway through.
    """
    from trajectory_guard_v04 import TrajectoryGuard  # type: ignore[import-not-found]

    dt = 0.005  # τ_LJ (stable window is ~0.01)
    masses = [1.0, 1.0]  # reduced units

    # Start at r = 1.15 σ (slightly compressed from LJ minimum r=2^(1/6)
    # σ ≈ 1.122σ) and give atoms opposite tiny velocities so the
    # dimer oscillates.
    r = [[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]
    v = [[-0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]

    F, E_pot = _lj_force_and_energy(r)

    def kinetic(v_: list) -> float:
        return 0.5 * sum(
            masses[i] * (v_[i][0] ** 2 + v_[i][1] ** 2 + v_[i][2] ** 2)
            for i in range(2)
        )

    E_kin = kinetic(v)
    E_total_initial = E_kin + E_pot

    frames = []
    save_every = max(1, n_steps // 100)

    guard = None
    if conserve is not None and tolerance is not None:
        guard = TrajectoryGuard(quantity=conserve, tolerance=tolerance)

    def snapshot(step: int):
        frame = _LJFrame(
            step=step,
            time=step * dt,
            positions=[row[:] for row in r],
            velocities=[row[:] for row in v],
            masses=masses[:],
            E_kinetic=kinetic(v),
            E_potential=E_pot,
        )
        frames.append(frame)
        if guard is not None:
            guard.sample(frame)

    # Initial sample
    snapshot(0)

    for step in range(1, n_steps + 1):
        # Half-kick
        for i in range(2):
            for k in range(3):
                v[i][k] += 0.5 * dt * F[i][k] / masses[i]
        # Drift
        for i in range(2):
            for k in range(3):
                r[i][k] += dt * v[i][k]
        # New forces
        F, E_pot = _lj_force_and_energy(r)
        # Second half-kick
        for i in range(2):
            for k in range(3):
                v[i][k] += 0.5 * dt * F[i][k] / masses[i]

        if step % save_every == 0 or step == n_steps:
            snapshot(step)

    E_total_final = frames[-1].E_total
    energy_drift = abs(E_total_final - E_total_initial)

    return {
        "frames": frames,
        "energy_drift": energy_drift,
        "E_initial": E_total_initial,
        "E_final": E_total_final,
        "n_steps": n_steps,
        "dt": dt,
    }


# Map: domain name -> callable(raw_kwarg_asts, line, col) that
# inspects the un-evaluated kwargs and raises QLangSyntaxError if any
# pre-evaluation invariant fails.  Called by the evaluator before it
# resolves identifiers inside the simulate block.
_PRE_VALIDATORS: Dict[str, Callable[..., None]] = {}


def _validate_md_kwargs(
    raw_kwargs: Tuple[Tuple[str, Any], ...], line: Optional[int], col: Optional[int]
) -> None:
    """Pre-evaluation validation for ``simulate md``.  Rejects a
    ``conserve: X`` where X is not a recognised conservation law."""
    for name, ast_node in raw_kwargs:
        if name != "conserve":
            continue
        # The AST node is an Identifier for bare names like
        # ``conserve: total_energy``.
        if type(ast_node).__name__ == "Identifier":
            ident = getattr(ast_node, "name", "")
            if ident not in _KNOWN_CONSERVED:
                known = ", ".join(sorted(_KNOWN_CONSERVED))
                raise QLangSyntaxError(
                    f"unknown conserve quantity {ident!r}; valid: {known}",
                    line=getattr(ast_node, "line", line),
                    col=getattr(ast_node, "col", col),
                )


def pre_validate(
    domain: str,
    raw_kwargs: Tuple[Tuple[str, Any], ...],
    *,
    line: Optional[int],
    col: Optional[int],
) -> None:
    """Entry point called by the evaluator before evaluating the
    simulate block's kwargs.  No-op for domains without a validator."""
    vr = _PRE_VALIDATORS.get(domain)
    if vr is not None:
        vr(raw_kwargs, line, col)


_PRE_VALIDATORS["md"] = _validate_md_kwargs


_register_defaults()
