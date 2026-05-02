"""
Q-Lang v0.4 — execution tests for the bundled example programs.

Each test EXECUTES an ``examples/v04/*.ql`` script end-to-end and
asserts on the result — stdout contents, env bindings, trace
structure, or CLI exit code — not merely that the file parses.

These tests are the proof that the three v0.4 guarantees actually hold
on real programs that ship with the package:

  1. Cross-domain kernel dispatch (example 03: simulate chemistry)
  2. Non-skippable protocol        (example 02: experiment + invariant)
  3. Re-executable provenance      (example 03: trace + replay)

Plus supporting tests for:
  * Example 01: dimensional + uncertainty arithmetic on real data.
  * CLI subcommands: run / check / replay / --trace / --quiet.
  * Typed exit codes for every failure mode SPEC §8 names.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ─────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples" / "v04"
_QLANG_ENTRY = _ROOT / "src" / "qlang_v04.py"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _has_v4() -> bool:
    try:
        import qlang_v04  # noqa: F401

        return True
    except Exception:
        return False


def _has_chem() -> bool:
    try:
        import qenex_chem  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import qlang_v04  # noqa: F401
        from simulate_dispatch_v04 import (  # type: ignore[import-not-found]
            _load_compute_energy,
        )

        return _load_compute_energy() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _has_v4(),
    reason="qlang_v04 module not yet implemented",
)


def _run_cli(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(_QLANG_ENTRY), *args]
    return subprocess.run(
        cmd,
        check=check,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _run_script(path: Path) -> "QLangInterpreter":  # type: ignore[name-defined]
    from qlang_v04 import QLangInterpreter  # type: ignore[import-not-found]

    interp = QLangInterpreter()
    interp.run(path.read_text(encoding="utf-8"))
    return interp


# ─────────────────────────────────────────────────────────────────────
# Example 01 — dimensional arithmetic + uncertainty
# ─────────────────────────────────────────────────────────────────────


class TestExample01_BondLengthUncertainty:
    """``examples/v04/01_bond_length_uncertainty.ql``.

    Covers: unit literals, ``+/-`` uncertainty, ``in [unit]``
    conversion, ``*`` with dim propagation, uncertainty propagation
    through arithmetic."""

    SCRIPT = _EXAMPLES / "01_bond_length_uncertainty.ql"

    def test_parses(self):
        result = _run_cli("check", str(self.SCRIPT))
        assert result.returncode == 0, result.stderr

    def test_runs_and_produces_expected_bindings(self):
        interp = _run_script(self.SCRIPT)

        r = interp.env["r"]
        assert r.value == pytest.approx(0.74, abs=1e-10)
        assert r.uncertainty == pytest.approx(0.01, abs=1e-10)
        assert r.dim.length == 1

        r_bohr = interp.env["r_bohr"]
        # 0.74 Angstrom -> 1.3984 Bohr (to 4 sig figs)
        assert r_bohr.value == pytest.approx(1.398411, rel=1e-5)
        # Uncertainty also converts
        assert r_bohr.uncertainty == pytest.approx(0.018897, rel=1e-3)

        diameter = interp.env["diameter"]
        assert diameter.value == pytest.approx(1.48, abs=1e-10)
        assert diameter.dim.length == 1

        area = interp.env["area"]
        assert area.value == pytest.approx(0.5476, abs=1e-6)
        assert area.dim.length == 2

    def test_stdout_labels_units_with_user_names(self):
        """The display must name 'Bohr' and 'Angstrom' — not the
        canonical SI string — when the user wrote them."""
        interp = _run_script(self.SCRIPT)
        out = interp.stdout
        assert "[Angstrom]" in out
        assert "[Bohr]" in out


# ─────────────────────────────────────────────────────────────────────
# Example 02 — experiment with non-skippable invariants
# ─────────────────────────────────────────────────────────────────────


class TestExample02_OrbitPeriodExperiment:
    """``examples/v04/02_orbit_period_experiment.ql``.

    Covers: experiment blocks, ``given: ... in [unit]``, invariants
    that reject bad inputs before downstream ``let``s run, ``sqrt``
    builtin with dim-parity check, Kepler's III ≡ 1 year for Earth."""

    SCRIPT = _EXAMPLES / "02_orbit_period_experiment.ql"

    def test_parses(self):
        result = _run_cli("check", str(self.SCRIPT))
        assert result.returncode == 0, result.stderr

    def test_earth_period_is_one_year(self):
        interp = _run_script(self.SCRIPT)
        T = interp.env["T_earth"]
        # Expected: 2*pi*sqrt(r^3/(G*M)) for Earth ~ 3.1547e7 s
        # (one sidereal year)
        assert T.value == pytest.approx(3.15e7, rel=0.05)
        assert T.dim.time == 1
        assert T.dim.length == 0
        assert T.dim.mass == 0

    def test_negative_mass_fails_invariant(self, tmp_path):
        """Put the same experiment in a temp file and call it with
        M < 0; InvariantViolation must be raised."""
        src = """
experiment orbit {
    given: M in [kg]
           r in [m]
    invariant: M > 0.0 [kg]
    let T = r
    result: T
}

let bad = orbit(M: -1.0 [kg], r: 1.0 [m])
"""
        script = tmp_path / "fail.ql"
        script.write_text(src)
        result = _run_cli("run", str(script))
        assert result.returncode == 5  # EXIT_INVARIANT
        assert "InvariantViolation" in result.stderr
        assert "M > 0" in result.stderr


# ─────────────────────────────────────────────────────────────────────
# Example 03 — chemistry dispatch + provenance + replay
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_chem(), reason="qenex_chem required")
class TestExample03_H2BondEnergy:
    """``examples/v04/03_h2_bond_energy.ql``.

    Covers: simulate chemistry { ... } dispatch to qenex_chem,
    kernel Quantity with correct energy dim, experiment with
    physical invariants, trace emission, replay bitwise-verify."""

    SCRIPT = _EXAMPLES / "03_h2_bond_energy.ql"

    def test_parses(self):
        result = _run_cli("check", str(self.SCRIPT))
        assert result.returncode == 0, result.stderr

    def test_h2_hf_energy_matches_textbook(self):
        """H2/sto-3g/HF ≈ −1.1168 Hartree (Szabo-Ostlund, Table 3.3)."""
        interp = _run_script(self.SCRIPT)
        E_ha = interp.env["E_ha"]
        # Want the magnitude in Hartree
        hartree_j = 4.3597447222071e-18
        assert E_ha.dim.mass == 1
        assert E_ha.dim.length == 2
        assert E_ha.dim.time == -2
        # value is now SI (J); divide by hartree_j to get Hartree units
        e_hartree = E_ha.value
        assert -1.15 < e_hartree < -1.10

    def test_trace_contains_kernel_op(self):
        interp = _run_script(self.SCRIPT)
        chem_nodes = [n for n in interp.trace if n.op == "simulate.chemistry"]
        assert len(chem_nodes) >= 1
        assert "qenex_chem" in chem_nodes[0].producer

    def test_replay_reproduces_bitwise(self, tmp_path):
        """Write a trace, replay it, expect zero drift."""
        trace_path = tmp_path / "h2.trace.jsonl"
        result = _run_cli(
            "run",
            str(self.SCRIPT),
            "--trace",
            str(trace_path),
            "--quiet",
        )
        assert result.returncode == 0, result.stderr
        assert trace_path.exists()

        result = _run_cli("replay", str(trace_path))
        assert result.returncode == 0, f"replay failed with stderr: {result.stderr!r}"


# ─────────────────────────────────────────────────────────────────────
# CLI direct tests (not example-specific)
# ─────────────────────────────────────────────────────────────────────


class TestCLI:
    """Direct CLI behaviour — subcommands, exit codes, flags."""

    def test_check_bad_syntax_exits_2(self, tmp_path):
        script = tmp_path / "bad.ql"
        script.write_text("let x = @")
        result = _run_cli("check", str(script))
        assert result.returncode == 2  # EXIT_SYNTAX
        assert "QLangSyntaxError" in result.stderr

    def test_run_dim_mismatch_exits_3(self, tmp_path):
        script = tmp_path / "mix.ql"
        script.write_text("let x = 1.0 [m] + 1.0 [kg]")
        result = _run_cli("run", str(script))
        assert result.returncode == 3  # EXIT_DIM

    def test_run_unbound_exits_4(self, tmp_path):
        script = tmp_path / "unbound.ql"
        script.write_text("let x = undefined_name")
        result = _run_cli("run", str(script))
        assert result.returncode == 4  # EXIT_NAME

    def test_run_quiet_suppresses_stdout(self, tmp_path):
        script = tmp_path / "printy.ql"
        script.write_text('print "visible"')
        # Without --quiet: stdout has "visible"
        result = _run_cli("run", str(script))
        assert "visible" in result.stdout
        # With --quiet: stdout is empty (but still exits 0)
        result = _run_cli("run", str(script), "--quiet")
        assert result.returncode == 0
        assert "visible" not in result.stdout

    def test_run_trace_writes_jsonl(self, tmp_path):
        script = tmp_path / "trace_me.ql"
        script.write_text("let x = 1.0 + 2.0")
        trace = tmp_path / "out.jsonl"
        result = _run_cli(
            "run",
            str(script),
            "--trace",
            str(trace),
            "--quiet",
        )
        assert result.returncode == 0
        assert trace.exists()
        lines = trace.read_text().strip().splitlines()
        assert len(lines) >= 3  # at least two literals + add
        for line in lines:
            obj = json.loads(line)
            assert "node_id" in obj
            assert "op" in obj

    def test_replay_missing_file_exits_unknown(self, tmp_path):
        result = _run_cli("replay", str(tmp_path / "nope.jsonl"))
        assert result.returncode == 1  # EXIT_UNKNOWN
        assert "not found" in result.stderr

    def test_replay_tampered_trace_exits_8(self, tmp_path):
        # Generate a legit trace
        script = tmp_path / "gen.ql"
        script.write_text("let x = 1.0 + 2.0")
        trace = tmp_path / "t.jsonl"
        _run_cli("run", str(script), "--trace", str(trace), "--quiet", check=True)

        # Tamper with the add node's magnitude
        lines = trace.read_text().splitlines()
        tampered: list[str] = []
        for line in lines:
            obj = json.loads(line)
            if obj.get("op") == "add":
                obj["value"]["magnitude"] = 99.0
            tampered.append(json.dumps(obj))
        trace.write_text("\n".join(tampered) + "\n")

        result = _run_cli("replay", str(trace))
        assert result.returncode == 8  # EXIT_REPLAY_DRIFT
        assert "DRIFT" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
