//! The content encoding of a constant ([`Emit`]).

use crate::{
    encoder::{ContentWriter, Emit},
    ir::{
        content::ContentSink,
        value::{ConstValue, DecryptedArrayData},
    },
    metadata::MetadataRef,
    target::Target,
};

impl<T: Target> Emit<T> for ConstValue<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        match self {
            Self::I8(field_0) => {
                Emit::emit(&0u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::I16(field_0) => {
                Emit::emit(&1u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::I32(field_0) => {
                Emit::emit(&2u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::I64(field_0) => {
                Emit::emit(&3u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::U8(field_0) => {
                Emit::emit(&4u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::U16(field_0) => {
                Emit::emit(&5u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::U32(field_0) => {
                Emit::emit(&6u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::U64(field_0) => {
                Emit::emit(&7u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::NativeInt(field_0) => {
                Emit::emit(&8u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::NativeUInt(field_0) => {
                Emit::emit(&9u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Float(field_0) => {
                Emit::emit(&10u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Vector(field_0) => {
                Emit::emit(&11u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::String(field_0) => {
                Emit::emit(&12u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::DecryptedString(field_0) => {
                Emit::emit(&13u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
            Self::Null => {
                Emit::emit(&14u16, writer)?;
                Ok(())
            }
            Self::True => {
                Emit::emit(&15u16, writer)?;
                Ok(())
            }
            Self::False => {
                Emit::emit(&16u16, writer)?;
                Ok(())
            }
            Self::Type(field_0) => {
                Emit::emit(&17u16, writer)?;
                writer.metadata(MetadataRef::TypeRef(field_0))?;
                Ok(())
            }
            Self::MethodHandle(field_0) => {
                Emit::emit(&18u16, writer)?;
                writer.metadata(MetadataRef::Method(field_0))?;
                Ok(())
            }
            Self::FieldHandle(field_0) => {
                Emit::emit(&19u16, writer)?;
                writer.metadata(MetadataRef::Field(field_0))?;
                Ok(())
            }
            Self::Symbol(field_0) => {
                Emit::emit(&20u16, writer)?;
                writer.metadata(MetadataRef::Symbol(field_0))?;
                Ok(())
            }
            Self::DecryptedArray(field_0) => {
                Emit::emit(&21u16, writer)?;
                Emit::emit(field_0, writer)?;
                Ok(())
            }
        }
    }
}

impl<T: Target> Emit<T> for DecryptedArrayData<T> {
    fn emit<S: ContentSink<T> + ?Sized>(
        &self,
        writer: &mut ContentWriter<'_, T, S>,
    ) -> Result<(), S::Error> {
        let Self {
            data,
            element_type_ref,
            element_size,
        } = self;
        Emit::emit(data, writer)?;
        writer.metadata(MetadataRef::TypeRef(element_type_ref))?;
        Emit::emit(element_size, writer)?;
        Ok(())
    }
}
