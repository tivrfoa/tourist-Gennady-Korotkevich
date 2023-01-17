// Benq solution
// https://codeforces.com/contest/1779/submission/187797314

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

    fn go(&mut self, roads: &[Vec<char>; 3], N: usize) {
        let n = N as i32;
        let mut ans = n;
        let mut x: i32 = n - 1;
        while x >= 0 && roads[2][x as usize] == '0' { x -= 1; }
        let mut y: i32 = n - 1;
        while y >= 0 && roads[0][y as usize] == '1' { y -= 1; }
        let mut z: i32 = n - 1;
        while z >= 0 && roads[1][z as usize] == '1' { z -= 1; }
        ans = ans.min(2 * (n - 1 - x));
        ans = ans.min(n - 1 - x + n - 1 - y);
        ans = ans.min(n - 1 - z + n - 1 - x);
        ans = ans.min(n - 1 - z + n - 1 - y);
        writeln!(self.out, "{ans}");
    }

    fn solve(&mut self) {
        let N: usize = self.scan.token();
        let mut roads: [Vec<char>; 3] = [vec![], vec![], vec![]];
        for i in 0..3 {
            roads[i] = self.scan.token::<String>().chars().collect();
        }
        dbg!(&roads);
        let bk = roads[0].len() - 1;
        if roads[0][bk] == roads[1][bk] && roads[1][bk] == roads[2][bk] {
            writeln!(self.out, "0");
            return;
        }
        loop {
            if roads[0][bk] == '1' && roads[1][bk] == '1' && roads[2][bk] == '0' {
                return self.go(&roads, N);
            }
            for t in roads.iter_mut() {
                for u in t.iter_mut() {
                    *u = if *u == '1' { '0' } else { '1' };
                }
            }
            if roads[0][bk] == '1' && roads[1][bk] == '1' && roads[2][bk] == '0' {
                return self.go(&roads, N);
            }
            roads.rotate_left(1);
        }
    }
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
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/hello2023/g.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/hello2023/g.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

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
