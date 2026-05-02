"""
Q-Lang v0.4 — command-line interface (SPEC §2.1 "qlang run" / "qlang replay").

Subcommands:

    qlang run   script.ql [--trace trace.jsonl] [--quiet]
        Parse + evaluate ``script.ql``.  With ``--trace PATH``,
        write the derivation DAG to JSONL.  Print stdout from the
        program unless ``--quiet`` is passed.  Exit 0 on success;
        exit code 2 for ``QLangSyntaxError`` (parse time), 3 for
        ``DimensionMismatchError``, 4 for ``UnboundNameError`` /
        ``RebindingError``, 5 for ``InvariantViolation``, 6 for
        ``ConservationViolation``, 7 for ``KernelError``, 1 for
        any other unexpected failure.  Zero-cost typed exit codes
        make CI integration simple.

    qlang replay  trace.jsonl
        Load a previously-emitted trace and verify it.  Exit 0 if
        every node reproduces bitwise; exit code 8 with a drift
        report printed to stderr otherwise.

    qlang check   script.ql
        Parse only (no evaluation).  Exit 0 if the program is
        syntactically valid under the v0.4 grammar; non-zero with a
        diagnostic otherwise.  Use in CI to gate .ql files.

The CLI wraps ``QLangInterpreter`` and ``replay()`` — no evaluation
logic lives here.  Keeping the CLI layer this thin means v0.4.1's
extended runtime semantics don't need a parallel CLI rewrite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from errors_v04 import (  # type: ignore[import-not-found]
    ConservationViolation,
    DimensionMismatchError,
    InvariantViolation,
    KernelError,
    QLangError,
    QLangSyntaxError,
    RebindingError,
    UnboundNameError,
)
from interp_v04 import QLangInterpreter  # type: ignore[import-not-found]
from parser_v04 import parse  # type: ignore[import-not-found]
from replay_v04 import replay as replay_trace  # type: ignore[import-not-found]


# Typed exit codes.  Stable contract; documented in qlang --help.
EXIT_OK = 0
EXIT_UNKNOWN = 1
EXIT_SYNTAX = 2
EXIT_DIM = 3
EXIT_NAME = 4
EXIT_INVARIANT = 5
EXIT_CONSERVATION = 6
EXIT_KERNEL = 7
EXIT_REPLAY_DRIFT = 8


def _exit_code_for(exc: BaseException) -> int:
    if isinstance(exc, QLangSyntaxError):
        return EXIT_SYNTAX
    if isinstance(exc, DimensionMismatchError):
        return EXIT_DIM
    if isinstance(exc, (UnboundNameError, RebindingError)):
        return EXIT_NAME
    if isinstance(exc, InvariantViolation):
        return EXIT_INVARIANT
    if isinstance(exc, ConservationViolation):
        return EXIT_CONSERVATION
    if isinstance(exc, KernelError):
        return EXIT_KERNEL
    if isinstance(exc, QLangError):
        return EXIT_UNKNOWN
    return EXIT_UNKNOWN


# ─────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> int:
    path = Path(args.script)
    if not path.is_file():
        print(f"qlang: file not found: {path}", file=sys.stderr)
        return EXIT_UNKNOWN

    source = path.read_text(encoding="utf-8")

    interp = QLangInterpreter()
    try:
        interp.run(source)
    except QLangError as e:
        print(f"qlang: {type(e).__name__}: {e}", file=sys.stderr)
        if args.trace:
            # Still emit the trace up to the failure \u2014 this is useful
            # for post-mortem debugging.
            try:
                interp.write_trace(args.trace)
            except Exception as te:
                print(
                    f"qlang: (additionally failed to write trace: {te})",
                    file=sys.stderr,
                )
        return _exit_code_for(e)
    except Exception as e:  # defensive: typed errors only, but be safe
        print(f"qlang: unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_UNKNOWN

    if not args.quiet and interp.stdout:
        sys.stdout.write(interp.stdout)

    if args.trace:
        interp.write_trace(args.trace)

    return EXIT_OK


def cmd_replay(args: argparse.Namespace) -> int:
    path = Path(args.trace)
    if not path.is_file():
        print(f"qlang: trace not found: {path}", file=sys.stderr)
        return EXIT_UNKNOWN

    report = replay_trace(path)
    if report.ok:
        if not args.quiet:
            print(f"qlang replay: OK ({path.name})")
        return EXIT_OK

    # Drift detected
    count = len(report.drifted_nodes)
    print(
        f"qlang replay: DRIFT detected in {count} node(s)",
        file=sys.stderr,
    )
    if report.first_mismatch is not None:
        fm = report.first_mismatch
        print(
            f"  first mismatch: node_id={fm.get('node_id', '?')[:16]} "
            f"op={fm.get('op', '?')}",
            file=sys.stderr,
        )
    return EXIT_REPLAY_DRIFT


def cmd_check(args: argparse.Namespace) -> int:
    path = Path(args.script)
    if not path.is_file():
        print(f"qlang: file not found: {path}", file=sys.stderr)
        return EXIT_UNKNOWN

    source = path.read_text(encoding="utf-8")
    try:
        parse(source)
    except QLangError as e:
        print(f"qlang check: {type(e).__name__}: {e}", file=sys.stderr)
        return _exit_code_for(e)

    if not args.quiet:
        print(f"qlang check: OK ({path.name})")
    return EXIT_OK


# ─────────────────────────────────────────────────────────────────────
# argparse plumbing
# ─────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qlang",
        description="Q-Lang v0.4 \u2014 scientific protocol language",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="execute a .ql script")
    p_run.add_argument("script", help="path to .ql script")
    p_run.add_argument(
        "--trace",
        metavar="PATH",
        help="write the derivation DAG to PATH (JSONL)",
    )
    p_run.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress program stdout",
    )
    p_run.set_defaults(func=cmd_run)

    p_replay = sub.add_parser(
        "replay",
        help="verify a previously-emitted trace reproduces bitwise",
    )
    p_replay.add_argument("trace", help="path to .jsonl trace")
    p_replay.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress success message",
    )
    p_replay.set_defaults(func=cmd_replay)

    p_check = sub.add_parser(
        "check",
        help="parse-check a .ql script without running it",
    )
    p_check.add_argument("script", help="path to .ql script")
    p_check.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="suppress success message",
    )
    p_check.set_defaults(func=cmd_check)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
