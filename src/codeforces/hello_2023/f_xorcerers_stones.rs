// Benq solution
// https://codeforces.com/contest/1779/submission/187777037

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
        let mut ans: Vec<usize> = vec![];
        let N: usize = self.scan.token();
        let mut ans: Vec<usize> = vec![];
        let mut dp: Vec<[bool; 32]> = vec![[false; 32]; N];
        let mut stor: Vec<[bool; 32]> = vec![[false; 32]; N];
        let mut adj: Vec<Vec<usize>> = vec![vec![]; N];
        let mut sub: Vec<u32> = vec![0; N];
        let mut A: Vec<usize> = self.scan.read_nums();

        for i in 1..N {
            let p = self.scan.token::<usize>() - 1;
            adj[i].push(p);
            adj[p].push(i);
        }
        solve1(&mut dp, &mut stor, &adj, &mut sub, &A, 0, usize::MAX);
        assert!(sub[0] == N as u32);
        if !dp[0][0] {
            writeln!(self.out, "-1");
            return;
        }
        reconstruct(&dp, &stor, &adj, &sub, &mut ans, &A, 0, 0, usize::MAX);
        ans.push(0);
        writeln!(self.out, "{}", ans.len());
        let ans: String = ans.into_iter().map(|n| (n + 1).to_string() + " ").collect();
        writeln!(self.out, "{}", ans.trim_end());
    }
}

fn reconstruct(dp: &[[bool; 32]],
        stor: &[[bool; 32]],
        adj: &[Vec<usize>],
        sub: &[u32],
        ans: &mut Vec<usize>,
        A: &[usize],
        mut v: usize, x: usize, p: usize) {

    assert!(dp[x][v]);
    if sub[x] % 2 == 0 && v == 0 {
        ans.push(x);
        return;
    }
    let mut child: Vec<usize> = vec![];
    for y in &adj[x] {
        if *y != p {
            child.push(*y);
        }
    }
    child.sort_by(|a, b| b.cmp(a));
    'child:
    for y in child {
        for a in 0..32 {
            if stor[y][a] && dp[y][v ^ a] {
                reconstruct(dp, stor, adj, sub, ans, A, v ^ a, y, x);
                v = a;
                continue 'child;
            }
        }
        unreachable!();
    }
    assert!(v == A[x]);
}

fn add(a: &[bool; 32], b: &[bool; 32]) -> [bool; 32] {
    let mut res = [false; 32];

    for i in 0..32 {
        for j in 0..32 {
            res[i ^ j] |= a[i] & b[j];
        }
    }

    res
}

fn solve1(dp: &mut [[bool; 32]],
        stor: &mut [[bool; 32]],
        adj: &[Vec<usize>],
        sub: &mut [u32],
        A: &[usize],
        x: usize, p: usize) {

    dp[x][A[x]] = true;
    sub[x] = 1;
    for y in &adj[x] {
        let y = *y;
        if y != p {
            solve1(dp, stor, adj, sub, A, y, x);
            sub[x] += sub[y];
            stor[y] = dp[x];
            dp[x] = add(&stor[y], &dp[y]);
        }
    }
    if sub[x] % 2 == 0 {
        dp[x][0] = true;
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
    let fr = File::open("tests_in_out/codeforces/hello2023/f.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/hello2023/f.out").unwrap();
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
