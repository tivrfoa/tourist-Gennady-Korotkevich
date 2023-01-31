// Original C++ solution: https://codeforces.com/blog/entry/111783

use std::slice::Iter;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, StdinLock, StdoutLock, Write},
    ops::{Add, Sub},
    str::{self, FromStr},
};

pub struct Solution<R, W: Write> {
    scan: UnsafeScanner<R>,
    out: BufWriter<W>,
}

impl<R: BufRead, W: Write> Solution<R, W> {
    fn new(reader: R, writer: W) -> Self {
        let scan = UnsafeScanner::new(reader);
        let out = BufWriter::new(writer);

        Self { scan, out }
    }

    fn solve(&mut self) {
        let n: usize = self.scan.token();
        let mut p: i32 = self.scan.token();
        p /= 10000;
        let mut C: Vec<Vec<i32>> = vec![vec![0; n + 1]; n + 1];
        for i in 0..=n {
            C[i][0] = 1;
            for j in 1..=i {
                C[i][j] = C[i - 1][j] + C[i - 1][j - 1];
            }
        }

        let mut dp: Vec<Vec<i32>> = vec![vec![0; n + 1]; n + 1];
        let mut aux: Vec<Vec<i32>> = vec![vec![0; n + 1]; n + 1];
        for b in 0..=n {
            (dp[0][b], aux[0][b]) = (1, 1);
        }
        for i in 1..=n {
            for b in 0..= n - i {
                for y in 0..= i - 1 {
                    dp[i][b] += C[i - 1][y] * aux[i - 1 - y][b] *
                        (dp[y][b + 1] * p + (if b == 0 { 0 } else { dp[y][b - 1] * (1 - p) }));
                }
                for j in 0..=i {
                    aux[i][b] += dp[j][b] * dp[i - j][b] * C[i][j];
                }
            }
        }
        let mut ans = dp[n][0];
        for i in (1..=2 * n as i32).step_by(2) {
            ans /= i;
        }

        writeln!(self.out, "{}", ans);
    }
}

const MOD: i32 = 998244353;


fn inverse(mut a: i32, mut m: i32) -> i32 {
    let (mut u, mut v) = (0, 1);
    while a != 0 {
        let t = m / a;
        m -= t * a; std::mem::swap(&mut a, &mut m);
        u -= t * v; std::mem::swap(&mut u, &mut v);
    }
    assert!(m == 1);
    u
}

#[allow(dead_code)]
fn get_input() -> String {
    let mut buf = String::new();
    io::stdin()
        .read_line(&mut buf)
        .expect("Failed to read stdin.");

    buf
}

#[allow(dead_code)]

pub struct UnsafeScanner<R> {
    reader: R,
    buf_str: Vec<u8>,
    buf_iter: str::SplitAsciiWhitespace<'static>,
}

impl<R: io::BufRead> UnsafeScanner<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            buf_str: vec![],
            buf_iter: "".split_ascii_whitespace(),
        }
    }

    pub fn read_line(&mut self) -> String {
        let mut buf = String::new();
        self.reader.read_line(&mut buf);

        buf
    }

    pub fn read_ints(&mut self) -> Vec<i32> {
        self.read_line()
            .split_ascii_whitespace()
            .map(|n| n.parse::<i32>().unwrap())
            .collect()
    }

    // pub fn read_nums<T: Add<Output=T> + Sub<Output=T> + Ord>(&mut self) -> T {
    pub fn read_nums<T: std::str::FromStr>(&mut self) -> Vec<T> {
        self.read_line()
            .split_ascii_whitespace()
            .map(|n| match n.parse::<T>() {
                Ok(n) => n,
                Err(_) => panic!("invalid number {n}"),
            })
            .collect()
    }

    pub fn token<T: str::FromStr>(&mut self) -> T {
        loop {
            if let Some(token) = self.buf_iter.next() {
                return token.parse().ok().expect("Failed parse");
            }
            self.buf_str.clear();
            self.reader
                .read_until(b'\n', &mut self.buf_str)
                .expect("Failed read");
            self.buf_iter = unsafe {
                let slice = str::from_utf8_unchecked(&self.buf_str);
                std::mem::transmute(slice.split_ascii_whitespace())
            }
        }
    }

    pub fn get_2_nums<T: str::FromStr>(&mut self) -> (T, T) {
        let a = self.token::<T>();
        let b = self.token::<T>();
        (a, b)
    }

    pub fn get_3_nums<T: str::FromStr>(&mut self) -> (T, T, T) {
        let a = self.token::<T>();
        let b = self.token::<T>();
        let c = self.token::<T>();
        (a, b, c)
    }

    pub fn get_4_nums<T: str::FromStr>(&mut self) -> (T, T, T, T) {
        let a = self.token::<T>();
        let b = self.token::<T>();
        let c = self.token::<T>();
        let d = self.token::<T>();
        (a, b, c, d)
    }
}

trait CharOp {
    fn sub(&self, other: Self) -> u8;
    fn addu8(&self, i: u8) -> char;
    fn addusize(&self, i: usize) -> char;
}

impl CharOp for char {
    fn sub(&self, other: Self) -> u8 {
        *self as u8 - other as u8
    }
    fn addu8(&self, i: u8) -> char {
        (*self as u8 + i) as char
    }
    fn addusize(&self, i: usize) -> char {
        (*self as u8 + i as u8) as char
    }
}

fn is_perfect_square(v: u64) -> bool {
    let u = (v as f64).sqrt().round() as u64;
    u * u == v
}

fn main() {
    let reader = io::stdin().lock();
    let writer = io::stdout().lock();
    let mut solution: Solution<StdinLock, StdoutLock> = Solution::new(reader, writer);

    // let t = solution.scan.token::<usize>();

    // for _ in 0..t {
        solution.solve();
    // }
}

#[test]
#[ignore]
fn test_interactive() {
    main();
}

#[test]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/round844/f.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/round844/f.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        solution.solve();
    }
}
