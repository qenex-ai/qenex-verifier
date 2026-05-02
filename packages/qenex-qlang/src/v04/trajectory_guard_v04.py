"""
Q-Lang v0.4 — runtime trajectory conservation guard (SPEC §3.3).

Samples a user-chosen conserved quantity at every saved frame of an
MD trajectory and raises ``ConservationViolation`` if the peak drift
from the initial value exceeds the declared ``tolerance:``.

Why this exists
---------------
The pre-existing ``conservation_analyzer.py`` is a STATIC analyser
over Q-Lang source text — it inspects the code for violation patterns
at parse time.  What SPEC §3.3 promises (and what a ``simulate md
{ conserve: total_energy, tolerance: ... }`` block needs) is a
RUNTIME guard: sample the conserved quantity AS THE TRAJECTORY RUNS
and abort if drift exceeds tolerance.  Those are complementary
checks; v0.4 ships both.

Supported conserved quantities (SPEC §3.3)
------------------------------------------

    total_energy      — E_kin + E_pot at each frame
    linear_momentum   — magnitude of Σ_i m_i v_i
    angular_momentum  — magnitude of Σ_i m_i (r_i × v_i), about origin
    charge            — Σ_i q_i  (exact conservation when no transfer)
    entropy           — k_B Σ_i ln(2 π e m_i k_B T / h²) — for an ideal
                         gas only; v0.4 raises NotImplementedError for
                         anything else (v0.5+ registers concrete
                         backends).  The name is recognised so the
                         pre-validator accepts it; trajectories that
                         try to enforce it at runtime get a typed error.

All other names are rejected at PRE-validation time in
``simulate_dispatch_v04._validate_md_kwargs`` (SPEC §3.3).

Public API
----------

    guard = TrajectoryGuard(
        quantity="total_energy",
        tolerance=1.0e-6,            # in whatever unit the quantity
                                      # has in the frames (e.g. Hartree,
                                      # kcal/mol) — match the MD kernel
                                      # you're sampling.
    )
    for frame in trajectory:
        guard.sample(frame)          # raises ConservationViolation if
                                      # peak |Δ| > tolerance

Access to the recorded drift curve:

    guard.values            # list[float] — one per sample
    guard.drift             # peak |value[i] - value[0]|
    guard.passed            # True iff drift <= tolerance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

from errors_v04 import ConservationViolation  # type: ignore[import-not-found]


# Names accepted at pre-validation AND at runtime.
_KNOWN_QUANTITIES = frozenset(
    {
        "total_energy",
        "linear_momentum",
        "angular_momentum",
        "charge",
    }
)

# Names accepted at pre-validation BUT raise at runtime until a
# concrete sampler is registered in v0.5+.
_ACCEPTED_AT_PARSE_ONLY = frozenset(
    {
        "entropy",
    }
)


@dataclass
class TrajectoryGuard:
    """Runtime drift monitor for one conserved quantity.

    Invariants:
      - ``self.values[0]`` is the reference value (first sample).
      - ``self.drift`` is ``max(|v_i - v_0|)`` over all samples so far.
      - Sampling stops as soon as drift exceeds tolerance — the guard
        raises ``ConservationViolation`` immediately, before the
        trajectory can accumulate further error.
    """

    quantity: str
    tolerance: float
    values: List[float] = field(default_factory=list)
    reference: Optional[float] = None

    def __post_init__(self) -> None:
        if self.quantity in _ACCEPTED_AT_PARSE_ONLY:
            raise NotImplementedError(
                f"conserve: {self.quantity!r} is a valid v0.4 name but "
                f"no runtime sampler is registered for it in v0.4.  "
                f"Scheduled for v0.5+."
            )
        if self.quantity not in _KNOWN_QUANTITIES:
            # Defensive: callers MUST pre-validate names via
            # simulate_dispatch._validate_md_kwargs before instantiating
            # this guard.  This is a programming-error assertion.
            raise ValueError(f"TrajectoryGuard: unknown quantity {self.quantity!r}")

    # -----------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------

    def sample(self, frame: Any) -> None:
        """Sample the declared conserved quantity from ``frame`` and
        check drift against tolerance.

        ``frame`` is expected to have the attributes produced by the
        v0.4 MD wrapper (see ``_call_md`` in simulate_dispatch_v04):

            frame.E_total       (total energy)
            frame.positions     (N, 3) ndarray
            frame.velocities    (N, 3) ndarray
            frame.masses        (N,)   ndarray
            frame.charges       (N,)   ndarray  (or absent)

        Missing attributes produce a clean ValueError naming what's
        needed — v0.4 forbids silent-return semantics.
        """
        value = _sample_quantity(self.quantity, frame)
        self.values.append(value)
        if self.reference is None:
            self.reference = value
            return
        drift = abs(value - self.reference)
        if drift > self.tolerance:
            raise ConservationViolation(
                quantity=self.quantity,
                drift=drift,
                tolerance=self.tolerance,
            )

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    @property
    def drift(self) -> float:
        if not self.values or self.reference is None:
            return 0.0
        return max(abs(v - self.reference) for v in self.values)

    @property
    def passed(self) -> bool:
        return self.drift <= self.tolerance


# ─────────────────────────────────────────────────────────────────────
# Per-quantity samplers
# ─────────────────────────────────────────────────────────────────────


def _sample_quantity(name: str, frame: Any) -> float:
    """Return the conserved quantity's value for ``frame``."""
    if name == "total_energy":
        E = getattr(frame, "E_total", None)
        if E is None:
            raise ValueError("TrajectoryGuard(total_energy): frame.E_total is required")
        return float(E)

    if name == "linear_momentum":
        V = getattr(frame, "velocities", None)
        M = getattr(frame, "masses", None)
        if V is None or M is None:
            raise ValueError(
                "TrajectoryGuard(linear_momentum): frame.velocities and "
                "frame.masses are required"
            )
        # p = Σ_i m_i v_i  — take magnitude
        px = sum(M[i] * V[i][0] for i in range(len(M)))
        py = sum(M[i] * V[i][1] for i in range(len(M)))
        pz = sum(M[i] * V[i][2] for i in range(len(M)))
        return math.sqrt(px * px + py * py + pz * pz)

    if name == "angular_momentum":
        V = getattr(frame, "velocities", None)
        R = getattr(frame, "positions", None)
        M = getattr(frame, "masses", None)
        if V is None or R is None or M is None:
            raise ValueError(
                "TrajectoryGuard(angular_momentum): frame.{positions, "
                "velocities, masses} are required"
            )
        Lx = sum(M[i] * (R[i][1] * V[i][2] - R[i][2] * V[i][1]) for i in range(len(M)))
        Ly = sum(M[i] * (R[i][2] * V[i][0] - R[i][0] * V[i][2]) for i in range(len(M)))
        Lz = sum(M[i] * (R[i][0] * V[i][1] - R[i][1] * V[i][0]) for i in range(len(M)))
        return math.sqrt(Lx * Lx + Ly * Ly + Lz * Lz)

    if name == "charge":
        Q = getattr(frame, "charges", None)
        if Q is None:
            # Zero-charge system: charge is trivially conserved.
            return 0.0
        return float(sum(Q))

    raise ValueError(f"_sample_quantity: unknown quantity {name!r}")
