//! A variable's definition site is metadata, not a fact about the current IR.
//!
//! An edit can leave it stale — pointing at a block that was renumbered, or an
//! instruction index that now holds something else. Dereferencing one without
//! confirming the variable really is defined there reads an unrelated
//! instruction's operands and rewrites against them, which is silent wrong code
//! rather than a missed optimization.
//!
//! `SsaFunction::recorded_definition` is the one place that dereference is
//! written, and it carries the confirmation (`op.defs().any(|d| d == var)`).
//! This test keeps it that way.

use std::{
    fs,
    path::{Path, PathBuf},
};

/// Functions permitted to read a definition site and fetch an instruction in
/// the same body, each with the reason the pairing is safe there.
///
/// Entries are `(file, function, reason)`. The list is deliberately short: a
/// new entry means someone is dereferencing a def site outside the guarded
/// lookup, and has to say why that is sound.
const ALLOWED: &[(&str, &str, &str)] = &[
    (
        "src/ir/function/queries.rs",
        "recorded_definition",
        "the guarded lookup itself: it confirms `op.defs()` contains the variable \
         before returning, and is what every other site is expected to call",
    ),
    (
        "src/ir/function/editor.rs",
        "replace_uses_checked",
        "dereferences a *use* site, not a definition site, and re-checks \
         `uses_var` before trusting it",
    ),
    (
        "src/ir/function/editor.rs",
        "can_replace_instruction_use_with_dominators",
        "reads the definition site's block and instruction index for ordering \
         and dominance comparisons only; the instruction it fetches is the use",
    ),
];

/// Every `.rs` file under `root`, recursively.
fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Splits a file into function-sized chunks, keyed by the function's name.
///
/// Crude on purpose: a chunk starting at each `fn` is enough to ask "does this
/// body both read a definition site and fetch an instruction", which is the
/// shape in question.
fn functions(text: &str) -> Vec<(String, String)> {
    let mut chunks: Vec<(String, String)> = Vec::new();
    let mut current_name = String::from("<file scope>");
    let mut current: Vec<&str> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim_start();
        let starts_fn = trimmed.starts_with("fn ")
            || trimmed.starts_with("pub fn ")
            || trimmed.starts_with("pub(crate) fn ")
            || trimmed.starts_with("pub(super) fn ")
            || trimmed.starts_with("const fn ")
            || trimmed.starts_with("pub const fn ");
        if starts_fn {
            chunks.push((current_name.clone(), current.join("\n")));
            current_name = trimmed
                .rsplit_once("fn ")
                .map(|(_, rest)| rest)
                .unwrap_or(trimmed)
                .split(['(', '<'])
                .next()
                .unwrap_or("?")
                .to_owned();
            current = Vec::new();
        }
        current.push(line);
    }
    chunks.push((current_name, current.join("\n")));
    chunks
}

#[test]
fn a_definition_site_is_dereferenced_only_through_the_guarded_lookup() {
    let mut offenders: Vec<String> = Vec::new();

    for path in rust_sources(Path::new("src")) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let file = path.to_string_lossy().replace('\\', "/");

        for (name, body) in functions(&text) {
            if !body.contains(".def_site()") || !body.contains(".instruction(") {
                continue;
            }
            let allowed = ALLOWED
                .iter()
                .any(|(allowed_file, allowed_fn, _)| *allowed_file == file && *allowed_fn == name);
            if !allowed {
                offenders.push(format!("{file}::{name}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} function(s) read a definition site and fetch an instruction outside the \
         guarded lookup. Call `SsaFunction::recorded_definition`, which confirms the \
         variable is defined there, or add an entry to ALLOWED saying why this one is \
         sound: {}",
        offenders.len(),
        offenders.join(", ")
    );
}

/// Every allowlist entry names a real function, and carries a reason.
///
/// Without this an entry outlives the code it excused and the list quietly
/// widens.
#[test]
fn every_allowlist_entry_is_live_and_justified() {
    for (file, function, reason) in ALLOWED {
        assert!(
            reason.len() > 30,
            "{file}::{function} needs a reason, not a placeholder"
        );
        let text = fs::read_to_string(Path::new(file)).unwrap_or_default();
        assert!(
            !text.is_empty(),
            "{file} does not exist; the allowlist entry for {function} is stale"
        );
        assert!(
            text.contains(&format!("fn {function}")),
            "{file} no longer defines {function}; remove the allowlist entry"
        );
    }
}
