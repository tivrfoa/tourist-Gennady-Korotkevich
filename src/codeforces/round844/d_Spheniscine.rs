// https://codeforces.com/contest/1782/submission/189701705

#![allow(dead_code, unused_macros, unused_imports)]
use std::{cell::{Cell, RefCell, UnsafeCell}, cmp::{Ordering, Reverse, max, min}, collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque, hash_map::{DefaultHasher, RandomState}}, error::Error, fmt::{Display, Write as FmtWrite}, hash::{BuildHasher, Hash, Hasher}, io::{BufWriter, Read, Stdin, Stdout, Write}, iter::{FromIterator, Peekable}, mem::swap, ops::*, process::exit, rc::Rc, str::{FromStr, from_utf8_unchecked}, time::{Duration, Instant}, convert::{TryInto, TryFrom}, marker::PhantomData};

const IO_BUF_SIZE: usize = 1 << 16;
type Input = Scanner<Stdin>;
type Output = BufWriter<Stdout>;
fn _init_input() -> Input { Scanner::new(std::io::stdin()) }
fn _init_output() -> Output { BufWriter::with_capacity(IO_BUF_SIZE, std::io::stdout()) }

#[repr(transparent)] struct Unsync<T>(T);
unsafe impl<T> Sync for Unsync<T> {}
 
type BadLazy<T> = Unsync<UnsafeCell<Option<T>>>;
impl<T> BadLazy<T> {
    const fn new() -> Self { Self(UnsafeCell::new(None)) }
}
 
static INPUT: BadLazy<Input> = BadLazy::new();
static OUTPUT: BadLazy<Output> = BadLazy::new();
 
fn inp<F: FnOnce(&mut Input) -> R, R>(f: F) -> R {
    unsafe { f((&mut *INPUT.0.get()).get_or_insert_with(_init_input)) }
}
fn out<F: FnOnce(&mut Output) -> R, R>(f: F) -> R {
    unsafe { f((&mut *OUTPUT.0.get()).get_or_insert_with(_init_output)) }
}

macro_rules! read {
    () => { read() };
    ($t: ty) => { read::<$t>() };
    ($t: ty, $($tt: ty),*) => { (read::<$t>(), $(read::<$tt>(),)*) };
    [$t: ty; $n: expr] => { read_vec::<$t>($n) };
}
macro_rules! println { 
    () => { out(|x| { let _ = writeln!(x); }) };
    ($exp: expr) => { out(|x| { let _ = writeln!(x, "{}", $exp); }) }; 
    ($fmt: expr, $($arg : tt )*) => { out(|x| { let _ = writeln!(x, $fmt, $($arg)*); }) }
}
macro_rules! print { 
    ($exp: expr) => { out(|x| { let _ = write!(x, "{}", $exp); }) }; 
    ($fmt: expr, $($arg : tt )*) => { out(|x| { let _ = write!(x, $fmt, $($arg)*); }) }
}

fn out_flush() { out(|x| { let _ = x.flush(); }); }

fn input_is_eof() -> bool { inp(|x| x.eof()) }
fn read_byte() -> u8 { inp(|x| x.byte()) }
fn read_bytes_no_skip(n: usize) -> Vec<u8> { inp(|x| x.bytes_no_skip(n)) }
fn read_bytes(n: usize) -> Vec<u8> { inp(|x| x.bytes(n)) }
fn read_bytes2(n: usize, m: usize) -> Vec<Vec<u8>> { inp(|x| x.bytes2(n, m)) }
fn read_token() -> Vec<u8> { inp(|x| x.token_bytes()) }
fn read_token_str() -> String { unsafe { String::from_utf8_unchecked(read_token()) } }
fn read_line() -> Vec<u8> { inp(|x| x.line_bytes()) }
fn read_line_str() -> String { unsafe { String::from_utf8_unchecked(read_line()) } }
fn read<T: FromStr>() -> T { read_token_str().parse::<T>().ok().expect("failed parse") }
fn read_vec<T: FromStr>(n: usize) -> Vec<T> { (0..n).map(|_| read()).collect() }
fn read_vec2<T: FromStr>(n: usize, m: usize) -> Vec<Vec<T>> { (0..n).map(|_| read_vec(m)).collect() }

struct Scanner<R: Read> {
    src: R,
    _buf: Vec<u8>,
    _pt: usize, // pointer
    _rd: usize, // bytes read
}

#[allow(dead_code)]
impl<R: Read> Scanner<R> {
    fn new(src: R) -> Scanner<R> {
        Scanner { src, _buf: vec![0; IO_BUF_SIZE], _pt: 1, _rd: 1 }
    }
 
    fn _check_buf(&mut self) {
        if self._pt == self._rd {
            self._rd = self.src.read(&mut self._buf).unwrap_or(0);
            self._pt = (self._rd == 0) as usize;
        }
    }
 
    // returns true if end of file
    fn eof(&mut self) -> bool {
        self._check_buf();
        self._rd == 0
    }
 
    // filters \r, returns \0 if eof
    fn byte(&mut self) -> u8 {
        loop {
            self._check_buf();
            if self._rd == 0 { return 0; }
            let res = self._buf[self._pt];
            self._pt += 1;
            if res != b'\r' { return res; }
        }
    }

    fn bytes_no_skip(&mut self, n: usize) -> Vec<u8> { (0..n).map(|_| self.byte()).collect() }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let res = self.bytes_no_skip(n);
        self.byte();
        res
    }
    fn bytes2(&mut self, n: usize, m: usize) -> Vec<Vec<u8>> { (0..n).map(|_| self.bytes(m)).collect() }
 
    fn token_bytes(&mut self) -> Vec<u8> {
        let mut res = Vec::new();
        let mut c = self.byte();
        while c <= b' ' {
            if c == b'\0' { return res; }
            c = self.byte();
        }
        loop {
            res.push(c);
            c = self.byte();
            if c <= b' ' { return res; }
        }
    }
 
    fn line_bytes(&mut self) -> Vec<u8> {
        let mut res = Vec::new();
        let mut c = self.byte();
        while c != b'\n' && c != b'\0' {
            res.push(c);
            c = self.byte();
        }
        res
    }
}

trait JoinToStr { 
    fn join_to_str(self, sep: &str) -> String;
    fn concat_to_str(self) -> String;
}
impl<T: Display, I: Iterator<Item = T>> JoinToStr for I { 
    fn join_to_str(mut self, sep: &str) -> String {
        match self.next() {
            Some(first) => {
                let mut res = first.to_string();
                while let Some(item) = self.next() {
                    res.push_str(sep);
                    res.push_str(&item.to_string());
                }
                res
            }
            None => { String::new() }
        }
    }
 
    fn concat_to_str(self) -> String {
        let mut res = String::new();
        for item in self { res.push_str(&item.to_string()); }
        res
    }
}
trait AsStr { fn as_str(&self) -> &str; }
impl AsStr for [u8] { fn as_str(&self) -> &str {std::str::from_utf8(self).expect("attempt to convert non-UTF8 byte string.")} }

macro_rules! veci {
    ($n:expr , $i:ident : $gen:expr) => {{
        let _veci_n = $n;
        let mut _veci_list = Vec::with_capacity(_veci_n);
        for $i in 0.._veci_n {
            _veci_list.push($gen);
        }
        _veci_list
    }};
    ($n:expr , $gen:expr) => { veci!($n, _veci_: $gen) }
}

fn abs_diff<T: Sub<Output = T> + PartialOrd>(x: T, y: T) -> T {
    if x < y { y - x } else { x - y }
}

trait CommonNumExt {
    fn div_ceil(self, b: Self) -> Self;
    fn div_floor(self, b: Self) -> Self;
    fn gcd(self, b: Self) -> Self;
    fn highest_one(self) -> Self;
    fn lowest_one(self) -> Self;
    fn sig_bits(self) -> u32;
}

macro_rules! impl_common_num_ext {
    ($($ix:tt = $ux:tt),*) => {
        $(
            impl CommonNumExt for $ux {
                fn div_ceil(self, b: Self) -> Self {
                    let q = self / b; let r = self % b;
                    if r != 0 { q + 1 } else { q }
                }
                fn div_floor(self, b: Self) -> Self { self / b }
                fn gcd(self, mut b: Self) -> Self {
                    let mut a = self;
                    if a == 0 || b == 0 { return a | b; }
                    let shift = (a | b).trailing_zeros();
                    a >>= a.trailing_zeros();
                    b >>= b.trailing_zeros();
                    while a != b {
                        if a > b { a -= b; a >>= a.trailing_zeros(); }
                        else { b -= a; b >>= b.trailing_zeros(); }
                    }
                    a << shift
                }
                #[inline] fn highest_one(self) -> Self { 
                    if self == 0 { 0 } else { const ONE: $ux = 1; ONE << self.sig_bits() - 1 } 
                }
                #[inline] fn lowest_one(self) -> Self { self & self.wrapping_neg() }
                #[inline] fn sig_bits(self) -> u32 { std::mem::size_of::<$ux>() as u32 * 8 - self.leading_zeros() }
            }

            impl CommonNumExt for $ix {
                fn div_ceil(self, b: Self) -> Self {
                    let q = self / b; let r = self % b;
                    if self ^ b >= 0 && r != 0 { q + 1 } else { q }
                }
                fn div_floor(self, b: Self) -> Self { 
                    let q = self / b; let r = self % b;
                    if self ^ b < 0 && r != 0 { q - 1 } else { q }
                }
                fn gcd(self, b: Self) -> Self {
                    fn w_abs(x: $ix) -> $ux { (if x.is_negative() { x.wrapping_neg() } else { x }) as _ }
                    w_abs(self).gcd(w_abs(b)) as _
                }
                #[inline] fn highest_one(self) -> Self { (self as $ux).highest_one() as _ }
                #[inline] fn lowest_one(self) -> Self { self & self.wrapping_neg() }
                #[inline] fn sig_bits(self) -> u32 { std::mem::size_of::<$ix>() as u32 * 8 - self.leading_zeros() }
            }
        )*
    }
}
impl_common_num_ext!(i8 = u8, i16 = u16, i32 = u32, i64 = u64, i128 = u128, isize = usize);

trait ChMaxMin<T> {
    fn chmax(&mut self, v: T) -> bool;
    fn chmin(&mut self, v: T) -> bool;
}
impl<T: PartialOrd> ChMaxMin<T> for Option<T> {
    fn chmax(&mut self, v: T) -> bool { if self.is_none() || v > *self.as_ref().unwrap() { *self = Some(v); true } else { false } }
    fn chmin(&mut self, v: T) -> bool { if self.is_none() || v < *self.as_ref().unwrap() { *self = Some(v); true } else { false } }
}
impl<T: PartialOrd> ChMaxMin<T> for T {
    fn chmax(&mut self, v: T) -> bool { if v > *self { *self = v; true } else { false } }
    fn chmin(&mut self, v: T) -> bool { if v < *self { *self = v; true } else { false } }
}

// * end commons * //

#[inline]
pub fn mul_mod_u32(a: u32, b: u32, m: u32) -> u32 {
    (a as u64 * b as u64 % m as u64) as u32
}

fn pow_mod_u32(a: u32, mut k: u64, m: u32) -> u32 {
    if m == 1 { return 0; }
    let mut a = a as u64;
    let m = m as u64;
    let mut r: u64 = 1;
    while k > 0 {
        if k & 1 == 1 {
            r = r * a % m;
        }
        k >>= 1;
        a = a * a % m;
    }
    r as u32
}

trait PrimeCheck {
    fn is_prime(self) -> bool;
}
impl PrimeCheck for u32 {
    fn is_prime(self) -> bool {
        const WITNESSES: [u32; 3] = [2, 7, 61];
        let n = self;
        if n < 64 { return (0x28208a20a08a28ac_u64 >> n) & 1 == 1 }; // mask encodes low primes
        if (0x1f75d77d >> (n % 30)) & 1 == 1 { return false; } // test for divisibility by 2, 3, 5

        let r = (n - 1).trailing_zeros();
        let d = n >> r;

        'a: for &a in WITNESSES.iter() {
            // if a % n == 0 { continue; }
            let mut x = pow_mod_u32(a, d as _, n);
            if x == 1 || x == n-1 { continue; }
            for _ in 0..r-1 {
                x = mul_mod_u32(x, x, n);
                if x == n-1 { continue 'a; }
            }
            return false;
        }

        true
    }
}

/// multiplicative inverse mod 2^32
fn mul_inv_u32(x: u32) -> u32 {
    debug_assert!(x & 1 == 1, "Only odd numbers have an inverse, got {}", x);
    let mut y = 1u32;
    for _ in 0..5 {
        y = y.wrapping_mul(2u32.wrapping_sub(x.wrapping_mul(y)));
    }
    y
}

fn pollards_rho_u32<R: RandomGen>(mut n: u32, rng: &R) -> Vec<u32> {
    let mut ans = Vec::<u32>::new();
    if n == 0 { return ans; }
    while n & 1 == 0 { 
        n >>= 1;
        ans.push(2);
    }
    if n == 1 { return ans; }
    let mut stk = vec![n];

    while let Some(n) = stk.pop() {
        if n.is_prime() {
            ans.push(n);
            continue;
        }

        let ni = mul_inv_u32(n.wrapping_neg());
        // Montgomery reduction, algorithm still works with x * 2^k mod n instead of x
        let rd = |x: u64| {
            let (z, c) = (ni.wrapping_mul(x as _) as u64 * n as u64).wrapping_sub((n as u64) << 32).overflowing_add(x);
            let z = (z >> 32) as u32;
            if c { z } else { z.wrapping_add(n) }
        };

        loop {
            let mut x = rng.gen_mod_u32(n);
            let c = rng.gen_mod_u32(n-1) + 1;
            let mut g = c.gcd(n);
            let mut q = 1u32;
            let mut xs = 0u32;
            let mut y = 0u32;
            let f = |x| { 
                rd(x as u64 * x as u64 + c as u64)
            };

            const ACC_REPS: i32 = 64;
            let mut l = 1;
            while g == 1 {
                y = x;
                for _ in 0..l-1 { x = f(x); }
                let mut k = 0;
                while k < l && g == 1 {
                    xs = x;
                    for _ in 0..min(ACC_REPS, l - k) {
                        x = f(x);
                        q = rd(q as u64 * abs_diff(x, y) as u64);
                    }
                    g = q.gcd(n);
                    k += ACC_REPS;
                }
                l <<= 1;
            }

            if g == n {
                loop {
                    xs = f(xs);
                    g = abs_diff(xs, y).gcd(n);
                    if g != 1 { break; }
                }
                if g == n { continue; }
            }

            stk.push(n / g);
            stk.push(g);
            break;
        }
    }
    ans
}

fn generate_seed() -> u64 {
    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    Instant::now().hash(&mut hasher);
    hasher.finish()
}

// https://github.com/tkaitchuck/Mwc256XXA64
#[derive(Debug, Clone)]
struct Mwc256XXA64 {
    state: Cell<[u64; 4]>
}
impl Mwc256XXA64 {
    pub fn from_seed(s0: u64, s1: u64) -> Self { 
        let res = Self { state: Cell::new([s0, s1, 0xcafef00dd15ea5e5, 0x14057B7EF767814F]) };
        for _ in 0..6 { res.gen_64(); }
        res
    }
    pub fn new() -> Self { Self::from_seed(generate_seed(), generate_seed()) }
}
impl RandomGen for Mwc256XXA64 {
    fn gen_64(&self) -> u64 {
        let [x1, x2, x3, c] = self.state.get();
        let t = (x3 as u128).wrapping_mul(0xfeb3_4465_7c0a_f413);
        let (low, hi) = (t as u64, (t >> 64) as u64);
        let res = (x3 ^ x2).wrapping_add(x1 ^ hi);
        let (x0, b) = low.overflowing_add(c);
        self.state.set([x0, x1, x2, hi.wrapping_add(b as u64)]);
        res
    }
    fn gen_32(&self) -> u32 { self.gen_64() as u32 }
}
type Rng = Mwc256XXA64;

pub trait RandomGen {
    fn gen_32(&self) -> u32;
    fn gen_64(&self) -> u64 {
        ((self.gen_32() as u64) << 32) | self.gen_32() as u64
    }
    fn gen_128(&self) -> u128 {
        ((self.gen_64() as u128) << 64) | self.gen_64() as u128
    }
    /// Generates a random `u32` in `0..n`.
    fn gen_mod_u32(&self, n: u32) -> u32 {
        let mut r = self.gen_32();
        let mut m = (r as u64) * (n as u64);
        let mut lo = m as u32;
        if lo < n {
            let t = n.wrapping_neg() % n;
            while lo < t {
                r = self.gen_32();
                m = (r as u64) * (n as u64);
                lo = m as u32;
            }
        }
        (m >> 32) as u32
    }
    fn gen_mod_u64(&self, n: u64) -> u64 {
        let mut r = self.gen_64();
        let mut m = (r as u128) * (n as u128);
        let mut lo = m as u64;
        if lo < n {
            let t = n.wrapping_neg() % n;
            while lo < t {
                r = self.gen_64();
                m = (r as u128) * (n as u128);
                lo = m as u64;
            }
        }
        (m >> 64) as u64
    }
    #[cfg(target_pointer_width = "32")]
    fn gen_mod_usize(&self, n: usize) -> usize {
        self.gen_mod_u32(n as u32) as usize
    }
    #[cfg(target_pointer_width = "64")]
    fn gen_mod_usize(&self, n: usize) -> usize {
        self.gen_mod_u64(n as u64) as usize
    }

    fn shuffle<T>(&self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            slice.swap(i, self.gen_mod_usize(i+1));
        }
    }

    fn f64(&self) -> f64 {
        const B: u32 = 64;
        const F: u32 = std::f64::MANTISSA_DIGITS - 1;
        f64::from_bits((1 << (B - 2)) - (1 << F) + (self.gen_64() >> (B - F))) - 1.0
    }

    fn i64(&self, l: i64, rx: i64) -> i64 {
        debug_assert!(rx > l);
        (l as u64 + self.gen_mod_u64((rx - l) as u64)) as i64
    }
}

fn divisors_u32(n: u32, rng: &Rng) -> Vec<u32> {
    let mut f = pollards_rho_u32(n, rng);
    f.sort();
    let mut res = vec![1];

    let m = f.len();
    let mut i = 0;
    while i < m {
        let p = f[i];
        let j = (i+1..m).find(|&j| f[j] != p).unwrap_or(m);
        let x = j - i;

        for i in 0..res.len() {
            let mut a = res[i];
            for _ in 0..x {
                a *= p;
                res.push(a);
            }
        }

        i = j;
    }

    res
} 

trait SqrtExt {
    fn sqrt_floor(self) -> Self;
    fn sqrt_ceil(self) -> Self;
    fn sqrt_round(self) -> Self;
}
impl SqrtExt for i64 {
    fn sqrt_floor(self) -> Self {
        let g = (self as f64).sqrt() as i64;
        if self < g * g { g - 1 } else { g }
    }

    fn sqrt_ceil(self) -> Self {
        let g = (self as f64).sqrt() as i64;
        if self > g * g { g + 1 } else { g }
    }

    fn sqrt_round(self) -> Self {
        let g = self.sqrt_floor();
        if g * g < self - g { g + 1 } else { g }
    }
}

#[allow(non_snake_case, non_upper_case_globals)]
fn main() {
    let num_cases: usize = read();
    let rng = Rng::new();
	
    for _case_num in 1..=num_cases {
        let n = read!(usize);
        let A = read![u32; n];

        let mut ans = 1;
        for i in 0..n { for j in i+1..n {
            let a = A[i]; let b = A[j];
            for p in divisors_u32(b - a, &rng) {
                let q = (b - a) / p;
                if p > q { continue; }
                if (p + q) & 1 == 0 {
                    let m = ((p + q) / 2) as i64;
                    let x = m * m - b as i64;
                    if x >= 0 {
                        ans.chmax(A.iter().filter(|&&a| {
                            let b = a as i64 + x;
                            let s = b.sqrt_floor();
                            s * s == b
                        }).count());
                    }
                }
            }
        }}

        println!(ans);
    }
 
    out_flush();
}