// Benq solution
// https://codeforces.com/contest/1779/submission/187755818

use std::{
    collections::HashSet,
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
        let mut ans: Vec<(usize, usize)> = Vec::with_capacity(n);
        let ones = vec!['1'; n];

        for i in 0..n {
            let mut s = ones.clone();
            s[i] = '0';
            let s = String::from_iter(s);
            writeln!(self.out, "? {} {}", i + 1, s);
            self.out.flush();
            let ret: usize = self.scan.token();
            ans.push((ret, i));
        }
        ans.sort_by(|a, b| b.cmp(a));
        let (mut sum1, mut sum2) = (0, 0);
        for i in 0..ans.len() {
            sum1 += ans[i].0;
            sum2 += ans.len() - 1 - i;
            if sum1 == sum2 {
                let mut ret = vec!['0'; n];
                for j in 0..i+1 {
                    ret[ans[j].1] = '1';
                }
                writeln!(self.out, "! {}", String::from_iter(ret));
                self.out.flush();
                return;
            }
        }
    }
}

fn main() {
    let reader = io::stdin().lock();
    let writer = io::stdout().lock();
    let mut solution: Solution<StdinLock, StdoutLock> = Solution::new(reader, writer);

    solution.solve();
}

#[test]
#[ignore]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/hello2023/e.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/hello2023/e.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        solution.solve();
    }
}

#[test]
fn test_interactive() {
    main();
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
fn get_2_nums<T: str::FromStr>(scan: &mut UnsafeScanner<io::StdinLock>) -> (T, T) {
    let a = scan.token::<T>();
    let b = scan.token::<T>();
    (a, b)
}

#[allow(dead_code)]
fn get_3_nums<T: str::FromStr>(scan: &mut UnsafeScanner<io::StdinLock>) -> (T, T, T) {
    let a = scan.token::<T>();
    let b = scan.token::<T>();
    let c = scan.token::<T>();
    (a, b, c)
}

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
}
