#!/usr/bin/env bash
#
# Checks that the manifest, the lockfile and the changelog agree.
#
# The rule is version-bumped-early, section-dated-late: `Cargo.toml` carries the
# version the open section will become, and dating that section happens only in
# the release commit, which contains nothing else. So:
#
#   while `[Unreleased]` has content
#       the manifest version is strictly greater than the newest dated section
#       and equal to Cargo.lock's own entry
#
#   on a tag ref
#       `[Unreleased]` is empty
#       the newest dated section equals the manifest version
#       and the tag is `v<version>`
#
# Usage:
#   scripts/check-release.sh            # development state
#   scripts/check-release.sh <tag>      # release state, e.g. v0.6.0

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

readonly CHANGELOG="CHANGELOG.md"
readonly MANIFEST="Cargo.toml"

fail() {
    printf 'release check: %s\n' "$*" >&2
    exit 1
}

# The `version = "..."` from the manifest's `[package]` section.
manifest_version() {
    awk '/^\[package\]/{p=1; next} /^\[/{p=0} p && /^version *=/{gsub(/[" ]/,"",$3); print $3; exit}' \
        "$MANIFEST"
}

# This crate's own version in Cargo.lock, or empty when the file is absent.
lock_version() {
    [ -f Cargo.lock ] || return 0
    awk '/^name = "analyssa"$/{found=1; next} found && /^version = /{gsub(/[" ]/,"",$3); print $3; exit}' \
        Cargo.lock
}

# The newest `## [x.y.z] - date` heading.
newest_dated_section() {
    grep -oE '^## \[[0-9]+\.[0-9]+\.[0-9]+\]' "$CHANGELOG" | head -1 | tr -d '#[] '
}

# Whether `## [Unreleased]` has any content before the next `## ` heading.
unreleased_has_content() {
    awk '/^## \[Unreleased\]/{u=1; next} u && /^## /{exit} u && NF {found=1} END {exit !found}' \
        "$CHANGELOG"
}

# `a > b` over dotted versions.
version_greater() {
    [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" = "$1" ]
}

main() {
    local tag="${1:-}"
    local version newest lock

    version="$(manifest_version)"
    [ -n "$version" ] || fail "no version in $MANIFEST"
    newest="$(newest_dated_section)"
    lock="$(lock_version)"

    if [ -n "$lock" ] && [ "$lock" != "$version" ]; then
        fail "Cargo.lock says $lock, $MANIFEST says $version"
    fi

    if [ -n "$tag" ]; then
        if unreleased_has_content; then
            fail "tag $tag, but [Unreleased] still has content -- date it first"
        fi
        [ "$newest" = "$version" ] ||
            fail "tag $tag: newest dated section is $newest, manifest is $version"
        [ "$tag" = "v$version" ] || fail "tag $tag does not match manifest version $version"
        printf 'release check: %s is consistent\n' "$tag"
        return 0
    fi

    unreleased_has_content ||
        fail "[Unreleased] is empty on a development tree -- either it is a release commit \
(pass the tag) or a change went undocumented"

    if [ -n "$newest" ] && ! version_greater "$version" "$newest"; then
        fail "manifest is $version but $CHANGELOG already has a dated [$newest]; \
the open section cannot become a version that is already released"
    fi

    printf 'release check: %s open over released %s\n' "$version" "${newest:-none}"
}

main "$@"
