pub use std::{
    cmp::Ordering,
    fmt::Display,
    io::{self, prelude::*, BufWriter, Stdout, StdoutLock},
    mem,
    num::Wrapping,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Bound, Div,
        DivAssign, Mul, MulAssign, RangeBounds, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign,
        Sub, SubAssign,
    },
    str::FromStr,
    time::SystemTime,
};

#[allow(unused_imports)]
pub use crate::{
    binary_search::*, ext::*, input::*, math::*, num_traits::*, ranges::*, recursive_function::*,
    vec2d::*,
};

unsafe fn solve<'a, I>(scan: &mut Scanner<'a, I>)
where
    I: Iterator<Item = &'a str>,
{
    let n = scan.u();
    let mut all_ones = "1".repeat(n);
    let mut counts = (0..n)
        .map(|i| {
            all_ones.as_mut_vec()[i] = b'0';
            outln!("? {} {}", i + 1, all_ones);
            flush!();
            all_ones.as_mut_vec()[i] = b'1';
            (scan.u(), i)
        })
        .vec();
    counts.sort_unstable();

    let mut masters = "0".repeat(n);
    masters.as_mut_vec()[counts.pop().unwrap().1] = b'1';
    for index in (0..counts.len()).rev() {
        let i = counts[index].1;
        outln!("? {} {}", i + 1, masters);
        flush!();
        if scan.u() > 0 {
            while counts.len() > index {
                masters.as_mut_vec()[counts.pop().unwrap().1] = b'1';
            }
        }
    }

    outln!("! {}", masters);
}

pub fn main() {
    unsafe {
        OUTPUT.0 = Some(io::stdout());
        OUTPUT.1 = Some(BufWriter::new(OUTPUT.0.as_ref().unwrap_unchecked().lock()));
    }

    unsafe {
        unsafe fn extend_lifetime<T: ?Sized>(reference: &T) -> &'static T {
            mem::transmute(reference)
        }
        let mut storage = String::new();
        let mut scan = Scanner::custom(
            None,
            io::stdin()
                .lines()
                .map(|l| {
                    storage = l.unwrap();
                    extend_lifetime(storage.as_str()).split_whitespace()
                })
                .flatten(),
        );
        solve(&mut scan);
    }

    unsafe {
        OUTPUT.1.take();
    }
}

static mut OUTPUT: (Option<Stdout>, Option<BufWriter<StdoutLock<'static>>>) = (None, None);
#[macro_export]
macro_rules! out {
    ($($arg:tt)*) => {
        unsafe {
            write!(OUTPUT.1.as_mut().unwrap_unchecked(), $($arg)*).unwrap();
        }
    };
}
#[macro_export]
macro_rules! outln {
    ($($arg:tt)*) => {
        unsafe {
            writeln!(OUTPUT.1.as_mut().unwrap_unchecked(), $($arg)*).unwrap();
        }
    };
}
#[macro_export]
macro_rules! flush {
    () => {
        unsafe {
            OUTPUT.1.as_mut().unwrap_unchecked().flush().unwrap();
        }
    };
}

// Template -------------------------------------------------------------------
pub mod input {
    use std::{
        iter::{FromIterator, Peekable},
        str::{self, FromStr, SplitWhitespace},
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum ScanError {
        Parse,
        Eof,
    }
    impl std::fmt::Display for ScanError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                ScanError::Parse => write!(f, "failed to parse token"),
                ScanError::Eof => write!(f, "tried to read past eof"),
            }
        }
    }
    impl std::error::Error for ScanError {}

    pub trait ToScanner {
        fn scanner(&self) -> Scanner<'_, SplitWhitespace<'_>>;
    }
    impl ToScanner for str {
        fn scanner(&self) -> Scanner<'_, SplitWhitespace<'_>> {
            Scanner::new(self)
        }
    }

    #[derive(Debug, Clone)]
    pub struct Scanner<'a, I: Iterator<Item = &'a str>> {
        buffer: Option<&'a str>,
        buffer_iter: Peekable<I>,
    }
    impl<'a> Scanner<'a, SplitWhitespace<'a>> {
        pub fn new(buffer: &'a str) -> Self {
            Self {
                buffer: Some(buffer),
                buffer_iter: buffer.split_whitespace().peekable(),
            }
        }
    }
    impl<'a, I: Iterator<Item = &'a str>> Scanner<'a, I> {
        /// # Safety
        ///
        /// If buffer is not None, then buffer_iter needs to iterate over
        /// slices of the memory that buffer points to. Otherwise it is
        /// undefined behaviour to call any of the read_all functions.
        pub fn custom(buffer: Option<&'a str>, buffer_iter: I) -> Self {
            Self {
                buffer,
                buffer_iter: buffer_iter.peekable(),
            }
        }

        pub fn read_all(mut self) -> &'a str {
            self.read_all_ref()
        }
        pub fn read_all_ref(&mut self) -> &'a str {
            // replace with self.buffer_iter.as_str() once that becomes stable
            // (on the iterators where that is available)
            unsafe {
                let buffer = self.buffer.unwrap();
                let start = buffer.as_ptr();
                let offset = self
                    .buffer_iter
                    .peek()
                    .map(|s| s.as_ptr().offset_from(start) as usize)
                    .unwrap_or(buffer.len());
                &buffer[offset..]
            }
        }

        pub fn peek(&mut self) -> Result<&'a str, ScanError> {
            self.buffer_iter.peek().copied().ok_or(ScanError::Eof)
        }
        pub fn eof(&mut self) -> bool {
            self.peek().is_ok()
        }

        /// The scan functions try to read the next item and return an error
        /// if you have already reached eof or if the item could not be parsed.
        pub fn scan(&mut self) -> Result<&'a str, ScanError> {
            self.buffer_iter.next().ok_or(ScanError::Eof)
        }
        pub fn scan_p<T: FromStr>(&mut self) -> Result<T, ScanError> {
            self.scan()?.parse().map_err(|_| ScanError::Parse)
        }
        pub fn scan_u(&mut self) -> Result<usize, ScanError> {
            self.scan_p::<usize>()
        }
        pub fn scan_i(&mut self) -> Result<i64, ScanError> {
            self.scan_p::<i64>()
        }
        pub fn scan_f(&mut self) -> Result<f64, ScanError> {
            self.scan_p::<f64>()
        }
        pub fn scan_s(&mut self) -> Result<String, ScanError> {
            self.scan_p::<String>()
        }
        pub fn scan_b(&mut self) -> Result<Vec<u8>, ScanError> {
            self.scan_s().map(|s| s.into_bytes())
        }

        /// The next functions continue to read items until they find one that can
        /// be parsed correctly, or return an eof error if it could not be found.
        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Result<&'a str, ScanError> {
            loop {
                match self.scan() {
                    Ok("") => continue,
                    value => return value,
                }
            }
        }
        pub fn next_p<T: FromStr>(&mut self) -> Result<T, ScanError> {
            loop {
                match self.scan_p() {
                    Err(ScanError::Parse) => continue,
                    value => return value,
                }
            }
        }
        pub fn next_u(&mut self) -> Result<usize, ScanError> {
            self.next_p::<usize>()
        }
        pub fn next_i(&mut self) -> Result<i64, ScanError> {
            self.next_p::<i64>()
        }
        pub fn next_f(&mut self) -> Result<f64, ScanError> {
            self.next_p::<f64>()
        }
        pub fn next_s(&mut self) -> Result<String, ScanError> {
            self.next_p::<String>()
        }
        pub fn next_b(&mut self) -> Result<Vec<u8>, ScanError> {
            self.next_s().map(|s| s.into_bytes())
        }

        /// The one-letter functions continue to read items until they find one
        /// that can be parsed correctly. Panics if eof is reached.
        pub fn n(&mut self) -> &'a str {
            self.next().unwrap()
        }
        pub fn p<T: FromStr>(&mut self) -> T {
            self.next_p().unwrap()
        }
        pub fn u(&mut self) -> usize {
            self.next_u().unwrap()
        }
        pub fn i(&mut self) -> i64 {
            self.next_i().unwrap()
        }
        pub fn f(&mut self) -> f64 {
            self.next_f().unwrap()
        }
        pub fn s(&mut self) -> String {
            self.next_s().unwrap()
        }
        pub fn b(&mut self) -> Vec<u8> {
            self.next_b().unwrap()
        }

        /// Collects the remaining tokens, ignoring any that cannot be parsed.
        pub fn collect<C, T>(self) -> C
        where
            C: FromIterator<T>,
            T: FromStr,
        {
            self.buffer_iter.filter_map(|v| v.parse().ok()).collect()
        }
        /// Tries to collect n tokens, ignoring any that cannot be parsed.
        /// If eof is reached, the tokens collected thus far will be returned.
        pub fn collect_n<C, T>(&mut self, n: usize) -> C
        where
            C: FromIterator<T>,
            T: FromStr,
        {
            (0..n).filter_map(|_| self.next_p().ok()).collect()
        }
        /// Collects exactly n tokens, ignoring any that cannot be parsed.
        /// Panics if eof is reached before collecting n tokens.
        pub fn collect_exactly_n<C, T>(&mut self, n: usize) -> C
        where
            C: FromIterator<T>,
            T: FromStr,
        {
            (0..n).map(|_| self.next_p().unwrap()).collect()
        }

        /// The vec functions work like their collect counterparts, but always
        /// collect into a Vec.
        pub fn vec<T: FromStr>(self) -> Vec<T> {
            self.collect()
        }
        pub fn vec_n<T: FromStr>(&mut self, n: usize) -> Vec<T> {
            self.collect_n(n)
        }
        pub fn vec_exactly_n<T: FromStr>(&mut self, n: usize) -> Vec<T> {
            self.collect_exactly_n(n)
        }
    }
}

pub mod num_traits {
    use std::ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub,
        SubAssign,
    };

    pub trait FromU8: Copy {
        fn from_u8(val: u8) -> Self;
    }
    macro_rules! from_u8_impl {
        ($($t: ident)+) => {$(
            impl FromU8 for $t {
                fn from_u8(val: u8) -> Self {
                    val as $t
                }
            }
        )+};
    }
    from_u8_impl!(i128 i64 i32 i16 i8 isize u128 u64 u32 u16 u8 usize f32 f64);

    pub trait ToF64: Copy {
        fn to_f64(self) -> f64;
    }
    macro_rules! to_f64_impl {
        ($($t: ident)+) => {$(
            impl ToF64 for $t {
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )+};
    }
    to_f64_impl!(i128 i64 i32 i16 i8 isize u128 u64 u32 u16 u8 usize f32 f64);

    pub trait Number:
        FromU8
        + ToF64
        + PartialOrd
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Rem<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
    {
    }
    macro_rules! number_impl {
        ($($t: ident)+) => {$(
            impl Number for $t {}
        )+};
    }
    number_impl!(i128 i64 i32 i16 i8 isize u128 u64 u32 u16 u8 usize f32 f64);

    pub trait Signed: Number + Neg<Output = Self> {}
    macro_rules! signed_impl {
        ($($t: ident)+) => {$(
            impl Signed for $t {}
        )+};
    }
    signed_impl!(i128 i64 i32 i16 i8 isize f32 f64);

    pub trait Unsigned: Number {}
    macro_rules! unsigned_impl {
        ($($t: ident)+) => {$(
            impl Unsigned for $t {}
        )+};
    }
    unsigned_impl!(u128 u64 u32 u16 u8 usize);

    pub trait Integer:
        Number
        + Ord
        + BitOr
        + BitAnd
        + BitXor
        + Shl
        + Shr
        + BitOrAssign
        + BitAndAssign
        + BitXorAssign
        + ShlAssign
        + ShrAssign
    {
    }
    macro_rules! integer_impl {
        ($($t: ident)+) => {$(
            impl Integer for $t {}
        )+};
    }
    integer_impl!(i128 i64 i32 i16 i8 isize u128 u64 u32 u16 u8 usize);

    pub trait Float: Number {}
    macro_rules! float_impl {
        ($($t: ident)+) => {$(
            impl Float for $t {}
        )+};
    }
    float_impl!(f32 f64);
}

pub mod ranges {
    use std::ops::{Range, RangeInclusive};

    use crate::{math, num_traits::Integer};

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct DiscreteRange<T> {
        pub start: T,
        pub end: T,
    }
    impl<T: Integer> DiscreteRange<T> {
        pub fn is_empty(&self) -> bool {
            self.start >= self.end
        }

        pub fn len(&self) -> T {
            if self.is_empty() {
                T::from_u8(0)
            } else {
                self.end - self.start
            }
        }

        pub fn contains(&self, x: T) -> bool {
            self.start <= x && self.end > x
        }

        pub fn is_singleton(&self) -> bool {
            self.start + T::from_u8(1) == self.end
        }

        pub fn is_splittable(&self) -> bool {
            self.start + T::from_u8(1) < self.end
        }

        pub fn is_disjoint(&self, other: Self) -> bool {
            self.is_empty()
                || other.is_empty()
                || self.start >= other.end
                || self.end <= other.start
        }

        pub fn is_subset(&self, other: Self) -> bool {
            self.is_empty()
                || (!other.is_empty() && self.start >= other.start && self.end <= other.end)
        }

        pub fn is_superset(&self, other: Self) -> bool {
            other.is_subset(*self)
        }

        pub fn midpoint(&self) -> Option<T> {
            if self.is_empty() {
                None
            } else {
                Some(math::midpoint(self.start, self.end - T::from_u8(1)))
            }
        }

        pub fn split(&self) -> Option<(Self, Self)>
        where
            Self: Sized,
        {
            if self.is_splittable() {
                let mid_plus_one =
                    T::from_u8(1) + math::midpoint(self.start, self.end - T::from_u8(1));
                Some((
                    Self {
                        start: self.start,
                        end: mid_plus_one,
                    },
                    Self {
                        start: mid_plus_one,
                        end: self.end,
                    },
                ))
            } else {
                None
            }
        }

        pub fn iter(&self) -> Range<T> {
            self.start..self.end
        }
    }
    impl<T: Integer> From<Range<T>> for DiscreteRange<T> {
        fn from(range: Range<T>) -> Self {
            Self {
                start: range.start,
                end: range.end,
            }
        }
    }
    impl<T: Integer> From<RangeInclusive<T>> for DiscreteRange<T> {
        fn from(range: RangeInclusive<T>) -> Self {
            Self {
                start: *range.start(),
                end: *range.end() + T::from_u8(1),
            }
        }
    }
}

pub mod ext {
    use std::{
        collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, LinkedList, VecDeque},
        hash::Hash,
    };

    use crate::num_traits::Float;

    #[allow(clippy::derive_partial_eq_without_eq)]
    #[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Hash)]
    pub struct OrdFloat<T>(pub T);
    impl<T: Float> Eq for OrdFloat<T> {}
    #[allow(clippy::derive_ord_xor_partial_ord)]
    impl<T: Float> Ord for OrdFloat<T> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.0.partial_cmp(&other.0).unwrap()
        }
    }

    pub trait IteratorExtensions {
        type Item;
        fn n(&mut self) -> Self::Item;
        fn vec(&mut self) -> Vec<Self::Item>;
        fn vec_deque(&mut self) -> VecDeque<Self::Item>;
        fn linked_list(&mut self) -> LinkedList<Self::Item>;
        fn array<const N: usize>(&mut self) -> [Self::Item; N];
    }
    impl<I: Iterator> IteratorExtensions for I {
        type Item = I::Item;
        fn n(&mut self) -> Self::Item {
            self.next().unwrap()
        }
        fn vec(&mut self) -> Vec<Self::Item> {
            self.collect()
        }
        fn vec_deque(&mut self) -> VecDeque<Self::Item> {
            self.collect()
        }
        fn linked_list(&mut self) -> LinkedList<Self::Item> {
            self.collect()
        }
        fn array<const N: usize>(&mut self) -> [Self::Item; N] {
            use std::convert::TryInto;
            self.vec().try_into().ok().unwrap()
        }
    }
    pub trait IteratorExtensionsHashSet {
        type Item;
        fn hash_set(&mut self) -> HashSet<Self::Item>;
    }
    impl<I> IteratorExtensionsHashSet for I
    where
        I: Iterator,
        I::Item: Eq + Hash,
    {
        type Item = I::Item;
        fn hash_set(&mut self) -> HashSet<Self::Item> {
            self.collect()
        }
    }
    pub trait IteratorExtensionsHashMap {
        type K;
        type V;
        fn hash_map(&mut self) -> HashMap<Self::K, Self::V>;
    }
    impl<I, K, V> IteratorExtensionsHashMap for I
    where
        I: Iterator<Item = (K, V)>,
        K: Eq + Hash,
    {
        type K = K;
        type V = V;
        fn hash_map(&mut self) -> HashMap<K, V> {
            self.collect()
        }
    }
    pub trait IteratorExtensionsBTreeSet {
        type Item;
        fn b_tree_set(&mut self) -> BTreeSet<Self::Item>;
    }
    impl<I> IteratorExtensionsBTreeSet for I
    where
        I: Iterator,
        I::Item: Ord,
    {
        type Item = I::Item;
        fn b_tree_set(&mut self) -> BTreeSet<Self::Item> {
            self.collect()
        }
    }
    pub trait IteratorExtensionsBTreeMap {
        type K;
        type V;
        fn b_tree_map(&mut self) -> BTreeMap<Self::K, Self::V>;
    }
    impl<I, K, V> IteratorExtensionsBTreeMap for I
    where
        I: Iterator<Item = (K, V)>,
        K: Ord,
    {
        type K = K;
        type V = V;
        fn b_tree_map(&mut self) -> BTreeMap<K, V> {
            self.collect()
        }
    }
    pub trait IteratorExtensionsBinaryHeap {
        type Item;
        fn binary_heap(&mut self) -> BinaryHeap<Self::Item>;
    }
    impl<I> IteratorExtensionsBinaryHeap for I
    where
        I: Iterator,
        I::Item: Ord,
    {
        type Item = I::Item;
        fn binary_heap(&mut self) -> BinaryHeap<Self::Item> {
            self.collect()
        }
    }
}

pub mod vec2d {
    use std::ops::{Index, IndexMut};

    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum Neighbor {
        #[default]
        Up,
        UpRight,
        Right,
        DownRight,
        Down,
        DownLeft,
        Left,
        UpLeft,
    }
    impl Neighbor {
        pub const ADJACENT: [Self; 4] = [Self::Up, Self::Right, Self::Down, Self::Left];
        pub const DIAGONALS: [Self; 4] =
            [Self::UpRight, Self::DownRight, Self::DownLeft, Self::UpLeft];
        pub const ALL: [Self; 8] = [
            Self::Up,
            Self::UpRight,
            Self::Right,
            Self::DownRight,
            Self::Down,
            Self::DownLeft,
            Self::Left,
            Self::UpLeft,
        ];

        pub fn offset(&self) -> [i64; 2] {
            match self {
                Self::Up => [-1, 0],
                Self::UpRight => [-1, 1],
                Self::Right => [0, 1],
                Self::DownRight => [1, 1],
                Self::Down => [1, 0],
                Self::DownLeft => [1, -1],
                Self::Left => [0, -1],
                Self::UpLeft => [-1, -1],
            }
        }

        /// Neighbor of. Returns the coordinates wrapped.
        pub fn of(&self, position: [i64; 2]) -> [i64; 2] {
            let offset = self.offset();
            [position[0] + offset[0], position[1] + offset[1]]
        }
    }

    #[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Vec2d<T>(pub Vec<Vec<T>>);
    impl<T> Vec2d<T> {
        pub fn len(&self) -> (usize, usize) {
            (
                self.0.len(),
                self.0.get(0).map(|r| r.len()).unwrap_or_default(),
            )
        }

        pub fn get(&self, index: [i64; 2]) -> Option<&T> {
            self.0.get(index[0] as usize)?.get(index[1] as usize)
        }
        pub fn get_mut(&mut self, index: [i64; 2]) -> Option<&mut T> {
            self.0
                .get_mut(index[0] as usize)?
                .get_mut(index[1] as usize)
        }

        pub fn get_row(&self, index: i64) -> Option<&Vec<T>> {
            self.0.get(index as usize)
        }
        pub fn get_row_mut(&mut self, index: i64) -> Option<&mut Vec<T>> {
            self.0.get_mut(index as usize)
        }
    }
    impl<T> Index<[i64; 2]> for Vec2d<T> {
        type Output = T;
        fn index(&self, index: [i64; 2]) -> &Self::Output {
            &self.0[index[0] as usize][index[1] as usize]
        }
    }
    impl<T> IndexMut<[i64; 2]> for Vec2d<T> {
        fn index_mut(&mut self, index: [i64; 2]) -> &mut Self::Output {
            &mut self.0[index[0] as usize][index[1] as usize]
        }
    }
    impl<T> Index<i64> for Vec2d<T> {
        type Output = Vec<T>;
        fn index(&self, index: i64) -> &Self::Output {
            &self.0[index as usize]
        }
    }
    impl<T> IndexMut<i64> for Vec2d<T> {
        fn index_mut(&mut self, index: i64) -> &mut Self::Output {
            &mut self.0[index as usize]
        }
    }
}

pub mod math {
    use std::{mem, ops::RangeFrom};

    use crate::num_traits::{Integer, Number, Unsigned};

    /// Rounds towards a if the type is an integer
    /// Risks overflowing if a and b have different signs
    pub fn midpoint<T>(a: T, b: T) -> T
    where
        T: Number,
    {
        a + (b - a) / T::from_u8(2)
    }

    pub fn gcd<T>(mut a: T, mut b: T) -> T
    where
        T: Integer + Unsigned,
    {
        while b != T::from_u8(0) {
            a %= b;
            mem::swap(&mut a, &mut b);
        }
        a
    }

    pub fn lcm<T>(a: T, b: T) -> T
    where
        T: Integer + Unsigned,
    {
        (a / gcd(a, b)) * b
    }

    pub fn pow_mod<T>(b: T, e: T, modulo: T) -> T
    where
        T: Integer + Unsigned,
    {
        if e == T::from_u8(0) {
            T::from_u8(1) % modulo
        } else {
            let half = pow_mod(b, e / T::from_u8(2), modulo);
            let combined = (half * half) % modulo;
            if e % T::from_u8(2) == T::from_u8(1) {
                (combined * b) % modulo
            } else {
                combined
            }
        }
    }

    pub fn is_prime<T>(n: T) -> bool
    where
        T: Integer + Unsigned,
        RangeFrom<T>: Iterator<Item = T>,
    {
        if n < T::from_u8(2) {
            false
        } else {
            for x in (T::from_u8(2)..).take_while(|&x| x * x <= n) {
                if n % x == T::from_u8(0) {
                    return false;
                }
            }
            true
        }
    }
}

pub mod binary_search {
    use crate::{math, num_traits::Integer, ranges::DiscreteRange};

    pub fn binary_search_first_true<T>(range: DiscreteRange<T>, mut p: impl FnMut(T) -> bool) -> T
    where
        T: Integer,
    {
        let mut low = range.start;
        let mut high = range.end;
        while low < high {
            let mid = math::midpoint(low, high);
            if p(mid) {
                high = mid;
            } else {
                low = mid + T::from_u8(1);
            }
        }
        high
    }

    pub fn partition_point<T>(range: DiscreteRange<T>, mut p: impl FnMut(T) -> bool) -> T
    where
        T: Integer,
    {
        binary_search_first_true(range, |i| !p(i))
    }
}

#[allow(clippy::too_many_arguments)]
pub mod recursive_function {
    use std::marker::PhantomData;

    macro_rules! recursive_function {
        ($name: ident, $trait: ident, ($($type: ident $arg: ident,)*)) => {
            pub trait $trait<$($type, )*Output> {
                fn call(&mut self, $($arg: $type,)*) -> Output;
            }

            pub struct $name<F, $($type, )*Output>
            where
                F: FnMut(&mut dyn $trait<$($type, )*Output>, $($type, )*) -> Output,
            {
                f: std::cell::UnsafeCell<F>,
                $($arg: PhantomData<$type>,
                )*
                phantom_output: PhantomData<Output>,
            }

            impl<F, $($type, )*Output> $name<F, $($type, )*Output>
            where
                F: FnMut(&mut dyn $trait<$($type, )*Output>, $($type, )*) -> Output,
            {
                pub fn new(f: F) -> Self {
                    Self {
                        f: std::cell::UnsafeCell::new(f),
                        $($arg: Default::default(),
                        )*
                        phantom_output: Default::default(),
                    }
                }
            }

            impl<F, $($type, )*Output> $trait<$($type, )*Output> for $name<F, $($type, )*Output>
            where
                F: FnMut(&mut dyn $trait<$($type, )*Output>, $($type, )*) -> Output,
            {
                fn call(&mut self, $($arg: $type,)*) -> Output {
                    unsafe { (&mut *self.f.get())(self, $($arg, )*) }
                }
            }
        }
    }

    recursive_function!(RecursiveFunction0, Callable0, ());
    recursive_function!(RecursiveFunction, Callable, (Arg arg,));
    recursive_function!(RecursiveFunction2, Callable2, (Arg1 arg1, Arg2 arg2,));
    recursive_function!(RecursiveFunction3, Callable3, (Arg1 arg1, Arg2 arg2, Arg3 arg3,));
    recursive_function!(RecursiveFunction4, Callable4, (Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4,));
    recursive_function!(RecursiveFunction5, Callable5, (Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5,));
    recursive_function!(RecursiveFunction6, Callable6, (Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6,));
    recursive_function!(RecursiveFunction7, Callable7, (Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7,));
    recursive_function!(RecursiveFunction8, Callable8, (Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8,));
    recursive_function!(RecursiveFunction9, Callable9, (Arg1 arg1, Arg2 arg2, Arg3 arg3, Arg4 arg4, Arg5 arg5, Arg6 arg6, Arg7 arg7, Arg8 arg8, Arg9 arg9,));
}

