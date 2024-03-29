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
        let mut r1: Vec<i32> = vec![0; n];
        let mut c1: Vec<i32> = vec![0; n];
        let mut r2: Vec<i32> = vec![0; n];
        let mut c2: Vec<i32> = vec![0; n];
        for i in 0..n {
            r1[i] = self.scan.token();
            c1[i] = self.scan.token::<i32>() - 1;
            r2[i] = self.scan.token();
            c2[i] = self.scan.token();
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|a, b| c1[*a].cmp(&c1[*b]));
        let mut s: BTreeSet<(i32, usize)> = BTreeSet::new();
        let mut ans = 0;
        let mut p1 = -1;
        let mut p2 = -1;
        for i in order {
            if r1[i] == 1 && r2[i] == 2 {
                if p1 >= c2[i] {
                    r1[i] = 2;
                }
                if p2 >= c2[i] {
                    r2[i] = 1;
                }
                if r1[i] > r2[i] {
                    continue;
                }
            }
            if r1[i] == 1 && r2[i] == 2 {
                while !s.is_empty() {
                    let it = s.last().unwrap().clone();
                    if it.0 >= c1[i] {
                        c2[it.1] = c1[i];
                        s.remove(&it);
                    } else {
                        break;
                    }
                }
                ans += (c2[i] - c1[i].max(p1)) + (c2[i] - c1[i].max(p2));
                p1 = c2[i];
                p2 = c2[i];
                s.insert((c2[i], i));
                continue;
            }
            assert!(r1[i] == r2[i]);
            if r1[i] == 1 {
                c1[i] = c1[i].max(p1);
                p1 = p1.max(c2[i]);
            } else {
                c1[i] = c1[i].max(p2);
                p2 = p2.max(c2[i]);
            }
            if c1[i] < c2[i] {
                ans += c2[i] - c1[i];
                s.insert((c2[i], i));
            }
        }

        writeln!(self.out, "{}", ans);
        for i in 0..n {
            c1[i] += 1;
            if r1[i] <= r2[i] && c1[i] <= c2[i] {
                writeln!(self.out, "{} {} {} {}", r1[i], c1[i], r2[i], c2[i]);
            } else {
                writeln!(self.out, "0 0 0 0");
            }
        }
    }
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

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        solution.solve();
    }
}

#[test]
#[ignore]
fn test_interactive() {
    main();
}

#[test]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/round844/e.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/round844/e.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        solution.solve();
    }
}
