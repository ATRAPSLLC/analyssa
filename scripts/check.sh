#!/usr/bin/env bash
#
# The crate's verification gates, in one place.
#
# CI invokes these same names, so a local run is the same run as CI and a
# command string cannot drift between the two. Every gate below mirrors a job in
# .github/workflows/ci.yml exactly.
#
# Usage:
#   scripts/check.sh <gate>...
#   scripts/check.sh all            every gate that currently exists
#   scripts/check.sh release-gate   the stable-only subset (no nightly, no coverage)
#
# Gates: fmt clippy test doc doctest deny coverage release

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

readonly ALL_GATES=(fmt clippy test doc doctest deny coverage release)

# The subset a stock stable toolchain can run: no nightly rustfmt, no cargo-deny,
# no cargo-llvm-cov. Still blocks a broken --all-features build and a failing
# docs.rs build (Cargo.toml sets all-features = true for docs.rs).
readonly RELEASE_GATES=(clippy test doc doctest release)

log() { printf '\n\033[1m== %s\033[0m\n' "$*" >&2; }

gate_fmt() {
    # Pinned to nightly on purpose: rustfmt.toml sets group_imports and
    # imports_granularity, which are nightly-only. A stable rustfmt silently
    # ignores unknown options and exits 0, so the unpinned form passes locally
    # and fails in CI -- exactly the drift one source of truth exists to prevent.
    cargo +nightly fmt --check
}

gate_clippy() {
    cargo clippy --all-targets --all-features -- -D warnings
}

gate_test() {
    # Both ways: the default build must stay serde-free, and the feature must
    # actually be exercised.
    cargo test
    cargo test --all-features
}

gate_doc() {
    RUSTDOCFLAGS="-D warnings -D missing-docs" \
        cargo doc --no-deps --all-features --document-private-items
}

gate_doctest() {
    RUSTDOCFLAGS="-D warnings" cargo test --doc --all-features
}

gate_deny() {
    cargo deny check
}

gate_release() {
    # Manifest, lockfile and changelog agree. On a tag ref, pass the tag so the
    # stricter release-state rules apply.
    scripts/check-release.sh "${GITHUB_REF_NAME:-}"
}

gate_coverage() {
    cargo llvm-cov --all-targets --all-features --fail-under-lines 70 --summary-only
}

run_gate() {
    local gate="$1"
    if ! declare -F "gate_${gate}" >/dev/null; then
        printf 'unknown gate: %s\n' "$gate" >&2
        return 2
    fi
    log "$gate"
    "gate_${gate}"
}

main() {
    if [ "$#" -eq 0 ]; then
        printf 'usage: %s <gate>... | all | release-gate\n' "$0" >&2
        return 2
    fi

    local -a gates=()
    for arg in "$@"; do
        case "$arg" in
            all) gates+=("${ALL_GATES[@]}") ;;
            release-gate) gates+=("${RELEASE_GATES[@]}") ;;
            *) gates+=("$arg") ;;
        esac
    done

    for gate in "${gates[@]}"; do
        run_gate "$gate"
    done

    log "OK: ${gates[*]}"
}

main "$@"
