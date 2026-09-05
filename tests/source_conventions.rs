//! Conventions the compiler does not check, enforced against the source tree.
//!
//! These are integration tests rather than library code: they read `src/` from
//! disk and assert properties of its text. Nothing here is part of the crate.

use std::{
    fs,
    path::{Path, PathBuf},
};

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

/// Imports belong in a file's import section, in test code as in production.
///
/// `AGENTS.md` requires it, and nothing else checks it: an import buried in a
/// test function is as invisible to a reader scanning a file's dependencies as
/// one buried anywhere else. The single relaxation is `use super::*;` at the
/// top of a `#[cfg(test)] mod tests`.
///
/// The check is indentation-based, which is exact for this crate's layout: a
/// module-level import sits at 0 or 4 spaces, so 8 or more means a function
/// body.
#[test]
fn no_import_sits_inside_a_function_body() {
    let mut offenders: Vec<String> = Vec::new();

    for root in ["src", "tests", "benches"] {
        for path in rust_sources(Path::new(root)) {
            let Ok(text) = fs::read_to_string(&path) else {
                continue;
            };
            for (number, line) in text.lines().enumerate() {
                let indent = line.len().saturating_sub(line.trim_start().len());
                if indent >= 8 && line.trim_start().starts_with("use ") {
                    offenders.push(format!("{}:{}", path.display(), number.saturating_add(1)));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} import(s) inside a function body; move them to the file's import section: {}",
        offenders.len(),
        offenders.join(", ")
    );
}

/// A file declares at most one `#[cfg(test)] mod`.
///
/// Splitting tests across several modules in one file means several import
/// sections to keep in agreement, and it hides which module a helper belongs
/// to. One module per file keeps `use super::*;` meaning one thing.
///
/// Test-only *items* outside a module are not affected: a `pub(crate)` helper
/// another file's tests call cannot live inside a private test module.
#[test]
fn a_file_declares_at_most_one_test_module() {
    let mut offenders: Vec<String> = Vec::new();

    for path in rust_sources(Path::new("src")) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let lines: Vec<&str> = text.lines().collect();
        let modules: Vec<String> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.trim_start() == "#[cfg(test)]")
            .filter_map(|(number, _)| {
                let next = lines.get(number.saturating_add(1))?.trim_start();
                next.strip_prefix("mod ")
                    .map(|rest| rest.trim_end_matches(" {").to_owned())
            })
            .collect();
        if modules.len() > 1 {
            offenders.push(format!(
                "{} declares {}",
                path.display(),
                modules.join(", ")
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "{} file(s) with more than one test module: {}",
        offenders.len(),
        offenders.join("; ")
    );
}

/// Production code addresses the crate through `crate::`, and names what it
/// imports.
///
/// `super::` ties a file's imports to where it sits rather than to what it
/// needs, and a glob over the crate's own modules makes its dependencies
/// unreadable — a reader cannot tell which module a name came from, and a new
/// export silently enters scope everywhere. Both are permitted at the top of a
/// `#[cfg(test)] mod tests`, where `use super::*;` is the point.
///
/// An external crate's prelude (`use rayon::prelude::*;`) is not covered: that
/// is how such a crate is documented to be used, and its contents are fixed by
/// a dependency rather than by this repository.
///
/// A file named `tests.rs` is a test module declared elsewhere as
/// `#[cfg(test)] mod tests;`, so it carries the relaxation without containing
/// the attribute itself.
#[test]
fn production_code_imports_by_crate_path_and_names_what_it_imports() {
    let mut offenders: Vec<String> = Vec::new();

    for path in rust_sources(Path::new("src")) {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };

        // Everything from the first `#[cfg(test)]` onward is test code.
        let production = text
            .lines()
            .take_while(|line| line.trim_start() != "#[cfg(test)]");

        for (number, line) in production.enumerate() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("use ") {
                continue;
            }
            let at = format!("{}:{}", path.display(), number.saturating_add(1));
            if trimmed.starts_with("use super") {
                offenders.push(format!("{at} reaches through `super`"));
            } else if trimmed.ends_with("::*;")
                && ["use crate::", "use self::"]
                    .iter()
                    .any(|internal| trimmed.starts_with(internal))
            {
                offenders.push(format!("{at} imports a glob over the crate's own modules"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} import(s) outside the convention: {}",
        offenders.len(),
        offenders.join(", ")
    );
}
