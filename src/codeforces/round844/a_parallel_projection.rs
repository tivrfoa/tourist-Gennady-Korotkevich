use std::{
    collections::HashSet,
        fs::File,
        io::{self, BufRead, BufReader, BufWriter, StdinLock, StdoutLock, Write},
        ops::{Add, Sub},
        str::{self, FromStr},
};
use std::slice::Iter;

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
        let (w, d, h) = self.scan.get_3_nums::<i32>();
        let (a, b, f, g) = self.scan.get_4_nums::<i32>();

        let east = f + h + a + (b - g).abs();
        let west = w - f + h + w - a + (b - g).abs();
        let south = g + h + b + (a - f).abs();
        let north = d - g + h + d - b + (a - f).abs();

        let ans = east.min(west.min(south.min(north)));

        writeln!(self.out, "{}", ans);
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
#[ignore]
fn test_interactive() {
    main();
}


#[test]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/round844/a.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/round844/a.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        solution.solve();
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
