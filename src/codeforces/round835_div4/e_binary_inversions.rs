use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, StdinLock, StdoutLock, Write},
    ops::{Add, Sub},
    str::{self, FromStr},
};

/*

We only need to check if it's better to:
  - don't flip anything;
  - change first 0 to 1;
  - change last 1 to 0.

I'll use zero suffix table to avoid timeout
*/

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
        let len: usize = self.scan.token();
        let nums: Vec<u8> = self.scan.read_nums();
        if len == 1 {
            writeln!(self.out, "{}", 0);
            return;
        }
        if len == 2 {
            if nums[0] == 0 && nums[1] == 1 {
                writeln!(self.out, "{}", 0);
            } else {
                writeln!(self.out, "{}", 1);
            }
            return;
        }
        let mut zeros_suffix: Vec<usize> = vec![0; len];

        for (i, _) in nums.iter().enumerate().rev() {
            if i < len - 1 {
                zeros_suffix[i] = zeros_suffix[i + 1];
            }
            zeros_suffix[i] += if nums[i] == 0 { 1 } else { 0 };
        }

        // dbg!(&zeros_suffix);

        let mut original_pairs: usize = 0;
        for i in 0..len - 1 {
            if nums[i] == 1 {
                original_pairs += zeros_suffix[i + 1];
            }
        }
         
        let mut max_pairs = original_pairs;

        // change first 0 to 1
        if let Some(idx) = nums.iter().position(|n| *n == 0) {
            if idx == len - 1 {
                // no point in change last 0 to 1
                // change len - 2 to 0 and return
                writeln!(self.out, "{}", (len - 2) * 2);
                return;
            }
            // count how many ones there are before idx
            let ones = nums.iter().take(idx + 1).filter(|n| **n == 1).count();
            let pairs = original_pairs - ones + zeros_suffix[idx + 1];
            if pairs > max_pairs {
                max_pairs = pairs;
            }
        } else {
            // no zeros, so just change last 1 to 0
            writeln!(self.out, "{}", len - 1);
            return;
        }

        // change last 1 to 0
        if let Some(idx) = nums.iter().rposition(|n| *n == 1) {
            if idx == 0 {
                let pairs = original_pairs - 1 + zeros_suffix[2];
                writeln!(self.out, "{}", pairs);
                return;
            }
            // count how many ones there are before idx
            let ones = nums.iter().take(idx).filter(|n| **n == 1).count();
            let to_remove = if idx + 1 == len { 0 } else { zeros_suffix[idx + 1] };
            let pairs = original_pairs + ones - to_remove;
            if pairs > max_pairs {
                max_pairs = pairs;
            }
        } else {
            // no ones, so just change first 0 to 1
            writeln!(self.out, "{}", len - 1);
            return;
        }

        writeln!(self.out, "{}", max_pairs);
    }
}

fn main() {
    let reader = io::stdin().lock();
    let writer = io::stdout().lock();
    let mut solution: Solution<StdinLock, StdoutLock> = Solution::new(reader, writer);
    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        let ans = solution.solve();
    }
}

#[test]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/round835_div4/e.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/round835_div4/e.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        let ans = solution.solve();
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
