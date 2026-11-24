//! Selecting lanes of vector constants: masks, compression, selects, shuffles,
//! permutes and gathers.

use crate::{
    ir::{
        integer::BitWidth,
        lanes::{MAX_LANE_BITS, VectorConst},
        ops::{VectorBitmaskKind, VectorMaskPrefix, VectorPredicateLayout},
    },
    target::VectorShuffleLane,
};

impl VectorConst {
    /// Counts selected true predicate bits and returns their first logical index.
    ///
    /// A missing first bit is represented by `u64::MAX`. The optional governing
    /// predicate has physical bit storage, without scalar broadcast. Invalid
    /// mask storage, spacing or insufficient source capacity is refused.
    #[must_use]
    pub fn mask_summary(
        &self,
        governing: Option<&Self>,
        length: u64,
        layout: VectorPredicateLayout,
        element_bits: u32,
    ) -> Option<(u64, u64)> {
        if self.lane_bits() != 1
            || governing.is_some_and(|mask| mask.lane_bits() != 1)
            || layout.bit(0, element_bits).is_none()
        {
            return None;
        }
        if length != 0 {
            let last = layout.bit(length.checked_sub(1)?, element_bits)?;
            if last >= u64::from(self.lanes())
                || governing.is_some_and(|mask| last >= u64::from(mask.lanes()))
            {
                return None;
            }
        }
        let mut count = 0u64;
        let mut first = u64::MAX;
        for index in 0..length {
            let bit = u32::try_from(layout.bit(index, element_bits)?).ok()?;
            if self.lane(bit) != 0 && governing.is_none_or(|mask| mask.lane(bit) == 1) {
                count = count.checked_add(1)?;
                first = first.min(index);
            }
        }
        Some((count, first))
    }

    /// Computes selected prefix predicate bits, clearing disabled and unused positions.
    ///
    /// Execution policies preserve or replace disabled destination bits separately.
    /// Source and governing predicates use the specified physical layout.
    #[must_use]
    pub fn mask_prefix(
        &self,
        kind: VectorMaskPrefix,
        governing: Option<&Self>,
        length: u64,
        layout: VectorPredicateLayout,
        element_bits: u32,
    ) -> Option<Self> {
        let (_, first) = self.mask_summary(governing, length, layout, element_bits)?;
        let mut result = Self::splat(1, self.lanes(), 0)?;
        for index in 0..length {
            let bit = u32::try_from(layout.bit(index, element_bits)?).ok()?;
            let enabled = governing.is_none_or(|mask| mask.lane(bit) == 1);
            let selected = match kind {
                VectorMaskPrefix::Before => index < first,
                VectorMaskPrefix::Including => index <= first,
                VectorMaskPrefix::Only => index == first,
            };
            result.set_lane(bit, u128::from(enabled && selected));
        }
        Some(result)
    }

    /// Returns the exclusive selected-bit prefix sum, wrapped to the result width.
    ///
    /// The index may equal the source capacity. Result widths from eight through
    /// 128 bits are supported. Byte-granular mask spacing uses the result width.
    #[must_use]
    pub fn mask_iota_value(
        &self,
        governing: Option<&Self>,
        index: u64,
        layout: VectorPredicateLayout,
        result_bits: u32,
    ) -> Option<u128> {
        if !(8..=MAX_LANE_BITS).contains(&result_bits) || !result_bits.is_power_of_two() {
            return None;
        }
        let (count, _) = self.mask_summary(governing, index, layout, result_bits)?;
        Some(BitWidth::new(result_bits).low_bits_u128(u128::from(count)))
    }

    /// Returns `self`'s lanes the `active` lanes select, packed or spread over
    /// `base`, which holds the other lanes: a compress (`vpcompressd`, SVE's
    /// `compact`) writes the selected lanes in order from lane 0, and an
    /// expand (`vpexpandd`) writes lane `i` of `self` to the `i`th selected
    /// lane. `None` where `active` or `base` does not have `self`'s lanes.
    #[must_use]
    pub fn packed(&self, active: &[bool], base: &Self, compress: bool) -> Option<Self> {
        self.computable()?;
        base.computable()?;
        let lanes = usize::try_from(self.lanes()).ok()?;
        if active.len() != lanes
            || base.lanes() != self.lanes()
            || base.lane_bits() != self.lane_bits()
        {
            return None;
        }
        let mut result = base.clone();
        let mut next = 0u32;
        for (index, selected) in (0u32..).zip(active) {
            if !*selected {
                continue;
            }
            if compress {
                result.set_lane(next, self.lane(index));
            } else {
                result.set_lane(index, self.lane(next));
            }
            next = next.checked_add(1)?;
        }
        Some(result)
    }

    /// Packs selected source elements into a low prefix and returns its length.
    ///
    /// The other result elements are zero placeholders; execution tail policy
    /// determines their architectural values. Invalid or insufficient predicate
    /// storage and data lanes wider than 128 bits are refused.
    #[must_use]
    pub fn compress_selected(
        &self,
        selector: &Self,
        length: u64,
        layout: VectorPredicateLayout,
    ) -> Option<(Self, u64)> {
        self.computable()?;
        if length > u64::from(self.lanes()) {
            return None;
        }
        selector.mask_summary(None, length, layout, self.lane_bits())?;
        let mut result = Self::splat(self.lane_bits(), self.lanes(), 0)?;
        let mut count = 0u32;
        for index in 0..length {
            let bit = u32::try_from(layout.bit(index, self.lane_bits())?).ok()?;
            if selector.lane(bit) != 0 {
                result.set_lane(count, self.lane(u32::try_from(index).ok()?));
                count = count.checked_add(1)?;
            }
        }
        Some((result, u64::from(count)))
    }

    /// Gathers elements through unsigned indices, returning zero outside source capacity.
    ///
    /// Indices can have a different width from data and are never narrowed to
    /// the data width. A one-element index vector broadcasts to all result lanes.
    /// Data and index lanes wider than [`MAX_LANE_BITS`] are refused.
    #[must_use]
    pub fn gather(&self, indices: &Self) -> Option<Self> {
        self.computable()?;
        indices.computable()?;
        if indices.lanes() != 1 && indices.lanes() != self.lanes() {
            return None;
        }
        let mut result = Self::zeroed(self.lane_bits(), self.lanes())?;
        for index in 0..self.lanes() {
            let selected = indices.lane(if indices.lanes() == 1 { 0 } else { index });
            let value = u32::try_from(selected)
                .ok()
                .filter(|source| *source < self.lanes())
                .map_or(0, |source| self.lane(source));
            result.set_lane(index, value);
        }
        Some(result)
    }

    /// Selects data through packed or byte-granular predicate bits.
    ///
    /// A set bit selects the right input. A scalar right value or scalar bit
    /// broadcasts. The predicate can hold unused positions beyond the data lanes.
    /// Data lanes wider than [`MAX_LANE_BITS`] are refused.
    #[must_use]
    pub fn select_mask(
        &self,
        right: &Self,
        mask: &Self,
        layout: VectorPredicateLayout,
    ) -> Option<Self> {
        self.computable()?;
        let bits = self.lane_bits();
        if right.lane_bits() != bits
            || right.lanes() != 1 && right.lanes() != self.lanes()
            || mask.lane_bits() != 1
        {
            return None;
        }
        let stride = match layout {
            VectorPredicateLayout::Packed => 1,
            VectorPredicateLayout::ByteGranular if bits.is_multiple_of(8) => bits.checked_div(8)?,
            VectorPredicateLayout::ByteGranular => return None,
        };
        let last = self.lanes().checked_sub(1)?.checked_mul(stride)?;
        if mask.lanes() != 1 && last >= mask.lanes() {
            return None;
        }
        let mut result = Self::zeroed(bits, self.lanes())?;
        for index in 0..self.lanes() {
            let selected = mask.lane(if mask.lanes() == 1 {
                0
            } else {
                index.checked_mul(stride)?
            }) != 0;
            let value = if selected {
                right.lane(if right.lanes() == 1 { 0 } else { index })
            } else {
                self.lane(index)
            };
            result.set_lane(index, value);
        }
        Some(result)
    }

    /// Returns the bits `kind` gathers from the lanes, lane 0's lowest: each
    /// lane's most significant bit, or each lane's lowest (an SVE predicate's
    /// bit per lane); `None` for more lanes than 128 bits hold.
    #[must_use]
    pub fn bitmask(&self, kind: VectorBitmaskKind) -> Option<u128> {
        if self.computable()?.lanes() > 128 {
            return None;
        }
        let bit = match kind {
            VectorBitmaskKind::LaneMostSignificantBits => self.lane_bits().saturating_sub(1),
            VectorBitmaskKind::PredicateBits => 0,
        };
        Some(
            (0u32..)
                .zip(self.lane_values())
                .fold(0u128, |mask, (index, lane)| {
                    mask | ((lane >> bit) & 1).checked_shl(index).unwrap_or(0)
                }),
        )
    }

    /// Returns the lane mask the mask bits in `self`'s lowest lane state, as an
    /// AVX-512 opmask holds them: `lanes` lanes of `lane_bits`, each all ones
    /// where its bit is set (bit `i` governing lane `i`) and zero where it is
    /// clear or past the lane's bits.
    #[must_use]
    pub fn lanes_of_bits(&self, lanes: u32, lane_bits: u32) -> Option<Self> {
        let bits = self.computable()?.lane(0);
        Self::from_lanes(
            lane_bits,
            (0..lanes).map(|lane| {
                if bits.checked_shr(lane).is_some_and(|bit| bit & 1 == 1) {
                    u128::MAX
                } else {
                    0
                }
            }),
        )
    }

    /// Returns each lane of `set` where the lane of `self`, the mask, has its
    /// most significant bit set, and of `clear` where it is clear, at the
    /// mask's lane width. Only the lanes chosen are read: an operand not
    /// known (`None`) folds wherever no lane of it is chosen.
    #[must_use]
    pub fn select(&self, clear: Option<&Self>, set: Option<&Self>) -> Option<Self> {
        let bits = self.computable()?.lane_bits();
        let relaid = |vector: Option<&Self>| -> Option<Option<Self>> {
            match vector {
                Some(vector) => {
                    let vector = vector.relaned(bits)?;
                    (vector.lanes() == self.lanes()).then_some(Some(vector))
                }
                None => Some(None),
            }
        };
        let (clear, set) = (relaid(clear)?, relaid(set)?);
        let lanes = (0..self.lanes())
            .map(|index| {
                let chosen = if (self.lane(index) >> bits.saturating_sub(1)) & 1 == 1 {
                    set.as_ref()
                } else {
                    clear.as_ref()
                };
                chosen.map(|vector| vector.lane(index))
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(bits, lanes)
    }

    /// Returns the lanes `selectors` pick from `self` and `right`, laid end to
    /// end: `Left(i)` lane `i` of `self`, `Right(i)` of `right`, `Zero` a zero
    /// lane. `None` for an undefined lane, a lane past its vector, or a
    /// `Right` with no `right`.
    #[must_use]
    pub fn shuffle(&self, right: Option<&Self>, selectors: &[VectorShuffleLane]) -> Option<Self> {
        self.computable()?;
        let lanes = selectors
            .iter()
            .map(|selector| match selector {
                VectorShuffleLane::Left(index) => {
                    (*index < self.lanes()).then(|| self.lane(*index))
                }
                VectorShuffleLane::Right(index) => {
                    let right = right?.relaned(self.lane_bits())?;
                    (*index < right.lanes()).then(|| right.lane(*index))
                }
                VectorShuffleLane::Zero => Some(0),
                VectorShuffleLane::Undef => None,
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(self.lane_bits(), lanes)
    }

    /// Returns the lanes the indices in `self` select from `tables`, laid end
    /// to end, the first lowest, at the tables' lane width; an index past
    /// them all folds nothing.
    #[must_use]
    pub fn permute(&self, tables: &[Self]) -> Option<Self> {
        let lane_bits = tables.first()?.computable()?.lane_bits();
        self.computable()?;
        let lanes = self
            .lane_values()
            .map(|index| {
                let mut select = u32::try_from(index).ok()?;
                for table in tables {
                    let table = table.relaned(lane_bits)?;
                    if select < table.lanes() {
                        return Some(table.lane(select));
                    }
                    select = select.wrapping_sub(table.lanes());
                }
                None
            })
            .collect::<Option<Vec<_>>>()?;
        Self::from_lanes(lane_bits, lanes)
    }
}
