//! The canonical content encoder: [`ContentWriter`] and the [`Emit`] trait
//! every encoded value implements, writing into a host's
//! [`ContentSink`] in the format
//! [`ir::content`](crate::ir::content) versions.

use std::marker::PhantomData;

use crate::{
    ir::{content::ContentSink, ops::FlagsMask, variable::SsaVarId},
    metadata::MetadataRef,
    target::Target,
};

/// Encodes fields while retaining the sink's exact refusal type.
pub struct ContentWriter<'a, T: Target, S: ContentSink<T> + ?Sized> {
    /// The caller's streaming consumer.
    sink: &'a mut S,
    /// Ties opaque references to the operation's target.
    target: PhantomData<T>,
}

impl<'a, T: Target, S: ContentSink<T> + ?Sized> ContentWriter<'a, T, S> {
    /// Borrows one consumer.
    pub fn new(sink: &'a mut S) -> Self {
        Self {
            sink,
            target: PhantomData,
        }
    }

    /// Visits borrowed metadata through the consumer.
    pub fn metadata(&mut self, value: MetadataRef<'_, T>) -> Result<(), S::Error> {
        self.sink.metadata(value)
    }

    /// Encodes optional borrowed metadata without allocating or losing absence.
    ///
    /// # Errors
    ///
    /// Returns the consumer's first refusal before visiting later metadata.
    pub fn optional_metadata(&mut self, value: Option<MetadataRef<'_, T>>) -> Result<(), S::Error> {
        match value {
            Some(value) => {
                1u8.emit(self)?;
                self.metadata(value)
            }
            None => 0u8.emit(self),
        }
    }

    /// Consumes existing payload storage, borrowed.
    fn raw_bytes(&mut self, bytes: &[u8]) -> Result<(), S::Error> {
        self.sink.bytes(bytes)
    }
}

/// Encodes one known data shape; exhaustive implementations cover every field.
pub trait Emit<T: Target> {
    /// Writes this value, returning the first refusal immediately.
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error>;

    /// Writes a collection's elements after its length, with no temporary collection.
    fn emit_elements<S: ContentSink<T> + ?Sized>(
        values: &[Self],
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error>
    where
        Self: Sized,
    {
        for value in values {
            value.emit(writer)?;
        }
        Ok(())
    }
}

macro_rules! integer_content {
    ($($integer:ty),* $(,)?) => { $(
        impl<T: Target> Emit<T> for $integer {
            fn emit<S: ContentSink<T> + ?Sized>(&self, writer: &mut ContentWriter<'_, T, S>) -> Result<(), S::Error> {
                writer.raw_bytes(&self.to_le_bytes())
            }
        }
    )* };
}

integer_content!(u16, u32, u64, u128, i8, i16, i32, i64, i128);

impl<T: Target> Emit<T> for u8 {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        writer.raw_bytes(&[*self])
    }

    fn emit_elements<S: ContentSink<T> + ?Sized>(
        values: &[Self],
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        writer.raw_bytes(values)
    }
}

impl<T: Target> Emit<T> for bool {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        u8::from(*self).emit(writer)
    }
}

impl<T: Target> Emit<T> for usize {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        (*self as u64).emit(writer)
    }
}

impl<T: Target> Emit<T> for str {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.len().emit(writer)?;
        writer.raw_bytes(self.as_bytes())
    }
}

impl<T: Target> Emit<T> for String {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.as_str().emit(writer)
    }
}

impl<T: Target, V: Emit<T> + ?Sized> Emit<T> for Box<V> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.as_ref().emit(writer)
    }
}

impl<T: Target, V: Emit<T>> Emit<T> for Option<V> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            None => 0u8.emit(writer),
            Some(value) => {
                1u8.emit(writer)?;
                value.emit(writer)
            }
        }
    }
}

impl<T: Target, V: Emit<T>> Emit<T> for [V] {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.len().emit(writer)?;
        V::emit_elements(self, writer)
    }
}

impl<T: Target, V: Emit<T>> Emit<T> for Vec<V> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.as_slice().emit(writer)
    }
}

impl<T: Target, V: Emit<T>, const N: usize> Emit<T> for [V; N] {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.as_slice().emit(writer)
    }
}

impl<T: Target, A: Emit<T>, B: Emit<T>> Emit<T> for (A, B) {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let (left, right) = self;
        left.emit(writer)?;
        right.emit(writer)
    }
}

impl<T: Target> Emit<T> for SsaVarId {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.index().emit(writer)
    }
}

impl<T: Target> Emit<T> for FlagsMask {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        self.bits().emit(writer)
    }
}
