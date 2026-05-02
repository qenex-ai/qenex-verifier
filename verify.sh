#!/bin/bash
# verify.sh — run all four reproducers, write JSON outputs, and verify
# their SHAs against MANIFEST.sha256.
#
# Usage: bash verify.sh
# Exit code: 0 if all four SHAs match, non-zero otherwise.
#
# This is the one-command verification gate referenced in the v2 paper
# abstract. A reviewer who runs this script gets a yes/no answer about
# whether their environment reproduces the published bit-level claims.

set -euo pipefail

cd "$(dirname "$0")"

echo "Running QENEX Verifier reproducers..."
echo "(Each script runs in --json mode and writes its output to a tmp file.)"
echo

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

for s in method_inventory module_inventory qlang_inventory precision_matrix; do
    OUT="$TMPDIR/qenex-verifier-claims-$s.json"
    echo "  $s ... "
    python3 "packages/qenex_chem/src/scripts/$s.py" --json > "$OUT" 2>/dev/null
done

echo
echo "Checking SHAs against MANIFEST.sha256..."
cd "$TMPDIR"
sha256sum -c "$OLDPWD/MANIFEST.sha256"

echo
echo "All four reproducer SHAs match the v2-paper-pinned manifest."
echo "Your environment reproduces the bit-level v2 claims."
