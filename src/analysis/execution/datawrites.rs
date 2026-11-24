//! The model of a routine that writes data but no code: what `getcwd`,
//! `fprintf` or `stat` may leave changed when it returns.
//!
//! Such a routine writes through the pointers it is handed, unless it only
//! reads what they point at, and writes data the program let escape: its own
//! state, a stream's buffer, `errno`. It writes no instruction, so every body
//! a run entered stays the code it was checked to be. What it writes through
//! a pointer is judged as any write is, so a buffer that held code changes
//! that code: `read(fd, buf, n)` into code is how a program loads code.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{analysis::execution::ModelCall, target::Target};

/// A call's arguments by position, as a set: bit `n` is argument `n`.
/// Arguments past the sixty-fourth are never members.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ArgumentSet(u64);

impl ArgumentSet {
    /// No argument.
    pub const EMPTY: Self = Self(0);

    /// The set with `argument` added; past the sixty-fourth, the set as it is.
    #[must_use]
    pub const fn with(self, argument: u8) -> Self {
        if argument < 64 {
            Self(self.0 | 1 << argument)
        } else {
            self
        }
    }

    /// Whether `argument` is a member.
    #[must_use]
    pub const fn contains(self, argument: usize) -> bool {
        argument < 64 && self.0 & 1 << argument != 0
    }

    /// The members as bits: bit `n` set where argument `n` is one.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Whether no argument is a member.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl FromIterator<u8> for ArgumentSet {
    fn from_iter<I: IntoIterator<Item = u8>>(arguments: I) -> Self {
        arguments.into_iter().fold(Self::EMPTY, Self::with)
    }
}

/// What a routine that writes data but no code may write
/// ([`LibraryModel::WritesData`](crate::analysis::execution::LibraryModel::WritesData)),
/// by its arguments. An argument no field names is a pointer it may write
/// through, which is what an argument it declares nothing of may be.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataWrites {
    /// The arguments it writes nothing through: pointers to what it only
    /// reads (`fprintf`'s format, `stat`'s path), and every argument that is
    /// no pointer (`getcwd`'s size).
    pub read_only: ArgumentSet,
    /// The pointers it declares to objects of a type that holds no code
    /// (`FILE *`, `struct stat *`): one whose value nobody knows points at
    /// data the program let escape, where any other unknown pointer may
    /// reach anything.
    pub typed: ArgumentSet,
    /// The argument holding its `printf` format, where it takes one. The
    /// arguments after it are written through only where the format names
    /// a `%n` conversion, or where nobody knows the format.
    pub format: Option<u8>,
    /// The argument holding the `va_list` the format's arguments come
    /// through (`vfprintf`), where they do not follow it. What a `%n` writes
    /// through such an argument nobody can place, so a format that may name
    /// one is a write that may reach anything.
    pub list: Option<u8>,
    /// The functions it is handed that it keeps without calling (`signal`'s
    /// handler): what becomes of each is that it is stored, so the call runs
    /// as this model ([`LibraryModel::accounts_for_callback`]), and nothing
    /// is written through it.
    ///
    /// [`LibraryModel::accounts_for_callback`]: crate::analysis::execution::LibraryModel::accounts_for_callback
    #[cfg_attr(feature = "serde", serde(default))]
    pub kept: ArgumentSet,
}

impl DataWrites {
    /// Forgets what the routine may have written: through each argument it
    /// may write through ([`ModelCall::forget_through`]), and the data the
    /// program let escape ([`ModelCall::forget_escaped_data`]). It may end
    /// the lifetime of what it reaches, as `fclose` ends its stream's
    /// buffers. Its result is unknown.
    pub fn write<T: Target>(self, call: &mut ModelCall<'_, T>) {
        let variadic_written = self.format.is_none_or(|format| {
            let limit = call.access.string_limit();
            call.pointer(usize::from(format))
                .and_then(|pointer| call.read_string(pointer, limit))
                .is_none_or(|format| printf_writes(&format))
        });
        if variadic_written && self.format.is_some() && self.list.is_some() {
            call.clobber();
            return;
        }
        for index in 0..call.argument_count() {
            if self.read_only.contains(index)
                || self.kept.contains(index)
                || self
                    .format
                    .is_some_and(|format| index > usize::from(format) && !variadic_written)
            {
                continue;
            }
            call.forget_through(index, self.typed.contains(index));
        }
        call.invalidate_lifetimes();
        call.forget_escaped_data();
    }
}

/// Whether the `printf` format `format` writes through one of the arguments
/// after it: whether it names a `%n` conversion, in any length (`%hhn`,
/// `%ln`, `%zn`), with any flags, width, precision or argument position.
#[must_use]
pub fn printf_writes(format: &[u8]) -> bool {
    let mut bytes = format.iter().copied().peekable();
    while let Some(byte) = bytes.next() {
        if byte != b'%' {
            continue;
        }
        // Everything up to the conversion: an argument position (`2$`),
        // flags, width, precision and length, none of them a conversion.
        while bytes
            .next_if(|byte| {
                byte.is_ascii_digit()
                    || matches!(
                        byte,
                        b'$' | b'-'
                            | b'+'
                            | b' '
                            | b'#'
                            | b'\''
                            | b'I'
                            | b'*'
                            | b'.'
                            | b'h'
                            | b'l'
                            | b'L'
                            | b'q'
                            | b'j'
                            | b'z'
                            | b'Z'
                            | b't'
                    )
            })
            .is_some()
        {}
        if bytes.next() == Some(b'n') {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_n_conversion_writes() {
        for (format, writes) in [
            (b"%s: %d\n".as_slice(), false),
            (b"100%% done".as_slice(), false),
            (b"%n".as_slice(), true),
            (b"%hhn".as_slice(), true),
            (b"%2$-08.3lln".as_slice(), true),
            (b"%zn and %s".as_slice(), true),
            (b"%%n".as_slice(), false),
            (b"%m".as_slice(), false),
            (b"trailing %".as_slice(), false),
        ] {
            assert_eq!(
                printf_writes(format),
                writes,
                "{}",
                String::from_utf8_lossy(format)
            );
        }
    }

    #[test]
    fn arguments_past_the_sixty_fourth_are_never_members() {
        let set: ArgumentSet = [0, 3, 63, 64, 200].into_iter().collect();
        assert!(set.contains(0) && set.contains(3) && set.contains(63));
        assert!(!set.contains(1) && !set.contains(64) && !set.contains(200));
    }
}
