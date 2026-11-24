#!/usr/bin/env bash
#
# Regenerates src/ir/ops/testdata/ops_golden.txt.
#
# `ops_behaviour_golden` compares against the checked-in file and, on a
# mismatch, prints the generated golden between two fences. This lifts that text
# out of the test output and writes it back, so the test itself stays hermetic:
# it reads nothing at runtime and writes nothing ever.
#
# Review the resulting diff before committing it; it is the record that a change
# to the operand matches was behaviour-preserving.
#
# Usage:
#   scripts/update-golden.sh

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

readonly GOLDEN="src/ir/ops/testdata/ops_golden.txt"
readonly BEGIN="----- BEGIN ops_golden.txt -----"
readonly END="----- END ops_golden.txt -----"

output="$(mktemp)"
generated="$(mktemp)"
trap 'rm -f "$output" "$generated"' EXIT

# The test fails exactly when there is something to write, so its status is
# not the script's.
cargo test --lib ops_behaviour_golden -- --nocapture --test-threads=1 >"$output" 2>/dev/null || true

if ! grep -qF -- "$BEGIN" "$output"; then
    if grep -q 'ops_behaviour_golden \.\.\. ok' "$output" \
        || grep -q 'test result: ok. 1 passed' "$output"; then
        printf '%s is already current\n' "$GOLDEN"
        exit 0
    fi
    printf 'ops_behaviour_golden did not run to a comparison; run it directly to see why:\n' >&2
    printf '  cargo test --lib ops_behaviour_golden\n' >&2
    exit 1
fi

awk -v begin="$BEGIN" -v end="$END" '
    !inside { at = index($0, begin); if (at) { inside = 1 } ; next }
    { at = index($0, end); if (at) { printf "%s", substr($0, 1, at - 1); exit } ; print }
' "$output" >"$generated"

mv "$generated" "$GOLDEN"
trap 'rm -f "$output"' EXIT
printf 'rewrote %s; review the diff:\n  git diff -- %s\n' "$GOLDEN" "$GOLDEN"
