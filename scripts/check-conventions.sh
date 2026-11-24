#!/usr/bin/env bash
#
# Conventions the compiler does not check, enforced against the source text.
#
# These are properties of how the source is written rather than of what it
# does, so they live here instead of in `cargo test`: the test suite stays
# hermetic (no test reads the tree from disk), and these still run on every CI
# push as the `conventions` gate.
#
# Checks:
#   imports-in-bodies   no `use` inside a function body (AGENTS.md, import style)
#   test-modules        a file declares at most one `#[cfg(test)] mod`
#   crate-paths         production code imports via `crate::`, never `super::`,
#                       and never globs the crate's own modules
#   editor-boundary     no pass mutates SSA through the raw `SsaFunction` helpers
#   hermetic-tests      no test reads the environment or the filesystem, spawns
#                       a process, or is `#[ignore]`d
#   present-tense-docs  no comment narrates the code's history
#   build-profile       no pass selects behaviour from the build profile
#   instruction-moves   a pass moves an instruction rather than rebuilding it
#   def-sites          a definition site is dereferenced only through the
#                       guarded lookup, `SsaFunction::recorded_definition`
#   derived-values      a pass creates a value from existing ones through the
#                       funnel that carries the host's facts across
#
# Usage:
#   scripts/check-conventions.sh

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failures=0

# Reports one failed check with its offenders, one per line.
report() {
    local check="$1" summary="$2" offenders="$3"
    [ -z "$offenders" ] && return 0
    printf '\n[%s] %s\n%s\n' "$check" "$summary" "$offenders" >&2
    failures=$((failures + 1))
}

# Every `.rs` file under the given roots, sorted, one per line.
rust_sources() {
    find "$@" -type f -name '*.rs' | LC_ALL=C sort
}

# Every root must exist: a missing one would make a check vacuous rather than
# failing it.
for root in src tests benches src/passes; do
    [ -d "$root" ] || { printf 'missing source root: %s\n' "$root" >&2; exit 1; }
done

# --- imports-in-bodies ------------------------------------------------------
#
# A `use` anywhere inside a function body, found by tracking brace depth from
# each `fn` to the brace that closes it. Indentation alone cannot decide it: a
# function body in an integration test sits at the same four spaces as a
# module-level import inside `mod tests`. Comments, strings and char literals
# are blanked first so a brace inside one does not move the depth.
report imports-in-bodies \
    "import(s) inside a function body; move them to the file's import section" \
    "$(rust_sources src tests benches | xargs awk '
        FNR == 1 { depth = 0; fns = 0; pending = 0; block_comment = 0 }
        {
            line = $0
            if (block_comment) { if (sub(/^.*\*\//, "", line)) block_comment = 0; else next }
            gsub(/\/\*.*\*\//, "", line)
            if (sub(/\/\*.*$/, "", line)) block_comment = 1
            gsub(/\\./, "", line)
            gsub(/"[^"]*"/, "\"\"", line)
            gsub(/'"'"'.'"'"'/, "", line)
            sub(/\/\/.*$/, "", line)
            trimmed = line; sub(/^[ \t]+/, "", trimmed)
            if (fns > 0 && trimmed ~ /^use /) print FILENAME ":" FNR ": " trimmed
            if (trimmed ~ /^(pub(\([a-z]+\))? )?((const|async|unsafe|extern "C") )*fn /) pending = 1
            if (pending == 1) square = 0
            n = length(line)
            for (i = 1; i <= n; i++) {
                c = substr(line, i, 1)
                if (c == "{") {
                    depth++
                    if (pending) { stack[++fns] = depth; pending = 0 }
                } else if (c == "}") {
                    if (fns > 0 && depth == stack[fns]) fns--
                    depth--
                } else if (c == "[") {
                    square++
                } else if (c == "]") {
                    square--
                } else if (c == ";" && pending && square == 0) {
                    # A bodiless declaration (trait method, extern fn); the
                    # `;` of an array type such as `[u8; 4]` does not count.
                    pending = 0
                }
            }
        }
    ')"

# --- test-modules -----------------------------------------------------------
#
# Several test modules in one file means several import sections to keep in
# agreement, and hides which module a helper belongs to. Test-only *items*
# outside a module are unaffected.
report test-modules \
    "file(s) declaring more than one #[cfg(test)] mod" \
    "$(rust_sources src | xargs awk '
        FNR == 1 { if (count > 1) print file ": " names; file = FILENAME; count = 0; names = ""; pending = 0 }
        {
            line = $0; sub(/^[ \t]+/, "", line)
            if (pending && line ~ /^mod /) {
                name = substr(line, 5); sub(/[ ;{].*$/, "", name)
                names = names (count ? ", " : "") name; count++
            }
            pending = (line == "#[cfg(test)]")
        }
        END { if (count > 1) print file ": " names }
    ')"

# --- crate-paths ------------------------------------------------------------
#
# Everything from a file's first `#[cfg(test)]` onward is test code, where
# `use super::*;` is the point. A file named `tests.rs` is a test module
# declared elsewhere, so it carries the relaxation without the attribute. An
# external crate's prelude glob is how such a crate is documented to be used,
# so only globs over `crate::` and `self::` count.
report crate-paths \
    "import(s) outside the convention (reach through super, or glob the crate's own modules)" \
    "$(rust_sources src | grep -v '/tests\.rs$' | xargs awk '
        FNR == 1 { production = 1 }
        {
            line = $0; sub(/^[ \t]+/, "", line)
            if (line == "#[cfg(test)]") production = 0
            if (!production || line !~ /^use /) next
            if (line ~ /^use super/) print FILENAME ":" FNR " reaches through super"
            else if (line ~ /::\*;$/ && line ~ /^use (crate|self)::/) print FILENAME ":" FNR " globs the crate'"'"'s own modules"
        }
    ')"

# --- editor-boundary --------------------------------------------------------
#
# Passes mutate SSA through `SsaFunction::edit` / `SsaEditor` only; the raw
# helpers skip the def-site and use-index maintenance the editor performs.
report editor-boundary \
    "SSA passes must mutate through SsaFunction::edit/SsaEditor only" \
    "$(find src/passes -maxdepth 1 -type f -name '*.rs' | LC_ALL=C sort | xargs awk '
        BEGIN {
            n = split("ssa.replace_uses_checked( ssa.propagate_copies( ssa.nop_copy_defining( " \
                      "ssa.replace_instruction_op( ssa.remove_instruction( ssa.repair_ssa( " \
                      "ssa.rebuild_ssa( ssa.strip_nops( ssa.compact_variables( ssa.block_mut( " \
                      "ssa.blocks_mut( ssa.variables_mut(", forbidden, " ")
        }
        FNR == 1 { production = 1 }
        production && index($0, "#[cfg(test)]") { production = 0 }
        production { for (i = 1; i <= n; i++) if (index($0, forbidden[i])) print FILENAME ":" FNR " uses " forbidden[i] }
    ')"

# --- hermetic-tests ---------------------------------------------------------
#
# A test depends on nothing outside the crate: no environment variable, no file
# read or written at runtime, no child process, and no test is `#[ignore]`d -
# doctests included (a doc example marked `ignore` or `no_run` is never run).
# `include_str!` is fine, since it is resolved at compile time. The library
# itself has no business with any of these either, so the scan covers `src/`
# whole as well as `tests/`.
#
# `env::` is matched through the std functions that reach outside the process
# rather than as a bare path: the crate has a module of its own called `env` -
# the per-function pass environment - and `scheduling::env::` is not
# `std::env::`.
report hermetic-tests \
    "environment, filesystem or process access, or an ignored test; a check over the source text belongs in this script, a golden in include_str!" \
    "$(rust_sources src tests | xargs grep -nE \
        '#\[ignore|cfg_attr\([^)]*ignore|//[/!]\s*```[a-z0-9_,]*(ignore|no_run)|\benv!\(|option_env!\(|\bstd::env\b|\benv::(var|vars|args|set_var|remove_var|current_dir|temp_dir)\b|\bstd::fs\b|\bfs::|\bFile::|Command::new|temp_dir\(' \
        || true)"

# --- present-tense-docs -----------------------------------------------------
#
# Comments describe the code as it is (AGENTS.md); `CHANGELOG.md` is where
# change is narrated. Only the unambiguous forms are matched - "is used to
# compute" is ordinary present tense, "this used to compute" is history - and
# references to another repository's gap tracker, which a reader of this crate
# cannot follow.
report present-tense-docs \
    "comment narrating history; describe the code as it is and keep the why (history belongs in CHANGELOG.md)" \
    "$(rust_sources src tests benches | xargs grep -nPi \
        '//.*(\b(it|this|that|which|these|they|what)\s+used to\b|\bpreviously (was|were|had|did|made|let|broke|paid)\b|\bGAPS [A-Z][0-9]|\bwas measured\b)' \
        || true)"

# --- build-profile ----------------------------------------------------------
#
# `cfg!(debug_assertions)` in an *expression* selects behaviour, and a pass that
# does so makes the optimized IR - and every listing, hash and recovered source
# taken from it - a function of how the compiler was invoked rather than of its
# input. `debug_assert!` and `#[cfg(test)]` are unaffected: neither decides what
# a pass emits.
#
# Scanned over the source rather than asserted at the call sites that once did
# it, because the property is that *nothing* does: a new pass branching on the
# profile breaks it exactly as the old ones did. A comment naming the macro
# describes behaviour rather than selecting any, so comment lines are skipped.
report build-profile \
    "expression(s) selecting behaviour from the build profile; the output of a pass is a function of its input alone" \
    "$(rust_sources src | xargs grep -n 'cfg!(debug_assertions)' \
        | grep -v ':[[:space:]]*//' \
        || true)"

# --- instruction-moves ------------------------------------------------------
#
# Rebuilding an instruction from its operation - `SsaInstruction::synthetic(op)`
# over an op cloned out of the IR - produces a *new* instruction: it loses the
# identity a host keys its per-instruction facts by, the host payload naming
# where the code came from, and the resolved result type. A pass relocating an
# instruction takes it out with `SsaEditor::detach_instruction` (or
# `move_instruction`) and re-inserts it, which keeps all three.
#
# Matched on the shape that carries an op straight from the IR into a synthetic
# instruction. Building a genuinely new operation from scratch is unaffected.
report instruction-moves \
    "pass(es) rebuild an instruction from a cloned op; move it with SsaEditor::detach_instruction so its identity, host payload and result type survive" \
    "$(rust_sources src/passes | xargs grep -nE \
        'SsaInstruction::synthetic\(([a-z_]+\.)?op(\(\))?\.clone\(\)\)' \
        || true)"

# --- derived-values ---------------------------------------------------------
#
# A value a pass computes from existing ones is those values under a new id: a
# renamed definition, a joined phi operand, a hoisted expression. What a host
# knows about the originals - a constant it proved, a place it tracked - has to
# come along, and `create_variable_derived` / `create_variable_joined` are the
# funnels that carry it. Bare `create_variable` states a value with no parent,
# and a host fact silently fails to reach it.
#
# Building fixture IR from nothing is the legitimate use and lives in test
# modules, which the `production` guard excludes. A pass that introduces a
# value no existing one stands for - a computation's new result, a constant -
# states so through `create_fresh_variable`, which this does not match.
#
# Scoped to the passes, where a derived value's parents are in hand at the
# point it is created. Reconstruction is not: it mints a phi result before the
# operands to join exist, and an entry placeholder that has no parent at all.
# Both are stated at their sites in `rebuild.rs`, and the join for the former
# runs later, in `SsaFunction::refresh_phi_facts`.
report derived-values \
    "pass(es) create a derived value through bare create_variable; use create_variable_derived or create_variable_joined so host facts follow it" \
    "$(rust_sources src/passes | xargs awk '
        FNR == 1 { production = 1 }
        production && index($0, "#[cfg(test)]") { production = 0 }
        production && /create_variable\(/ && !/create_variable_(derived|joined|for_origin)\(/ {
            print FILENAME ":" FNR
        }
    ')"

# --- carried-parallel -------------------------------------------------------
#
# A host's per-thread context, such as the account its allocations charge,
# follows an item of a parallel iterator onto the pool thread that runs it only
# through `parallel::MapCarried`. So every production parallel iterator's first
# step past any `with_max_len` is a carried map (`map_carried` or
# `map_init_carried`); anything else runs its items under whatever the taking
# thread had marked. `src/parallel.rs` defines the carrying itself. Whitespace
# is removed first, so a chain split over lines reads as one. A scan that finds
# no parallel iterator at all is reported too: it would pass vacuously.
report carried-parallel \
    "parallel iterator(s) whose first step is not a carried map; start the chain with map_carried or map_init_carried" \
    "$(rust_sources src | grep -v '^src/parallel\.rs$' | xargs awk '
        function check(token,    rest, after, at, end) {
            rest = text
            while ((at = index(rest, token)) > 0) {
                seen++
                rest = substr(rest, at + length(token))
                after = rest
                while (substr(after, 1, 14) == ".with_max_len(") {
                    end = index(after, ")")
                    after = substr(after, end + 1)
                }
                if (substr(after, 1, 13) != ".map_carried(" &&
                    substr(after, 1, 18) != ".map_init_carried(") {
                    print file ": " token substr(after, 1, 50)
                }
            }
        }
        function flush() {
            if (file != "") { check("par_iter()"); check("par_iter_mut()") }
        }
        FNR == 1 { flush(); file = FILENAME; text = ""; production = 1; previous = "" }
        production && previous == "#[cfg(test)]" && /^mod tests/ { production = 0 }
        production { line = $0; gsub(/[ \t]/, "", line); text = text line }
        { previous = $0 }
        END { flush(); if (seen == 0) print "no parallel iterator found under src/: the scan no longer reaches them" }
    ')"

# --- def-sites --------------------------------------------------------------
#
# A variable's definition site is metadata, not a fact about the current IR: an
# edit can leave it pointing at a renumbered block or at an index that now holds
# another instruction. `recorded_definition` confirms the variable really is
# defined there. A function body that both reads a def site and fetches an
# instruction is dereferencing one, and must be that lookup or be listed here
# with the reason it is sound.
#
# Entries are `file|function|reason`. A new entry means someone is
# dereferencing a def site outside the guarded lookup, and has to say why.
readonly DEF_SITE_ALLOWED="\
src/ir/function/queries.rs|recorded_definition|the guarded lookup itself: it confirms op.defs() contains the variable before returning
src/ir/function/editor.rs|replace_uses_checked|dereferences a use site, not a definition site, and re-checks uses_var before trusting it
src/ir/function/editor.rs|can_replace_instruction_use_with_dominators|reads the def site's block and index for ordering and dominance only; the instruction it fetches is the use
src/ir/function/rebuild.rs|verify|prints the def site in an undefined-use message without dereferencing it; the instruction it fetches is the reader at the reported use site"

# Every entry names a live function and carries a real reason, so the list
# cannot outlive the code it excused and quietly widen.
stale=""
while IFS='|' read -r file function reason; do
    if [ "${#reason}" -le 30 ]; then
        stale+="${file}::${function} needs a reason, not a placeholder"$'\n'
    elif [ ! -f "$file" ]; then
        stale+="${file} does not exist; the entry for ${function} is stale"$'\n'
    elif ! grep -qE "fn ${function}[<(]" "$file"; then
        stale+="${file} no longer defines ${function}; remove the entry"$'\n'
    fi
done <<<"$DEF_SITE_ALLOWED"
report def-sites-allowlist "stale allowlist entries" "${stale%$'\n'}"

# Chunks each file at every line that starts a function (crude on purpose: a
# chunk per `fn` is enough to ask whether one body does both things).
def_site_bodies="$(rust_sources src | xargs awk '
    function flush() { if (defsite && instr) print file "|" name; defsite = 0; instr = 0 }
    FNR == 1 { flush(); file = FILENAME; name = "<file scope>" }
    {
        line = $0; sub(/^[ \t]+/, "", line)
        if (line ~ /^(pub(\((crate|super)\))? )?(const )?fn /) {
            flush()
            name = line; sub(/^.*fn /, "", name); sub(/[(<].*$/, "", name)
        }
        if (index($0, ".def_site()")) defsite = 1
        if (index($0, ".instruction(")) instr = 1
    }
    END { flush() }
')"
unlisted=""
while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    if ! cut -d'|' -f1,2 <<<"$DEF_SITE_ALLOWED" | grep -qxF "$hit"; then
        unlisted+="${hit/|/::}"$'\n'
    fi
done <<<"$def_site_bodies"
report def-sites \
    "function(s) read a definition site and fetch an instruction outside the guarded lookup; call SsaFunction::recorded_definition, or add an allowlist entry saying why this one is sound" \
    "${unlisted%$'\n'}"

if [ "$failures" -ne 0 ]; then
    printf '\n%d convention check(s) failed\n' "$failures" >&2
    exit 1
fi
printf 'conventions: all checks passed\n'
