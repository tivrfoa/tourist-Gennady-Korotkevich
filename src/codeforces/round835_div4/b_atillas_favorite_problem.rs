use io::Write;
use std::io::prelude::*;
use std::{io::{self, StdinLock, StdoutLock, BufWriter, BufReader}, str, fs::File};

struct Solution<'a, R> {
    scan: UnsafeScanner<R>,
    out: BufWriter<StdoutLock<'a>>,
}

impl<R: io::BufRead> Solution<'_, R> {
    fn new(reader: R) -> Self {
        let mut scan = UnsafeScanner::new(reader);
        let mut out = io::BufWriter::new(io::stdout().lock());

        Self {
            scan,
            out
        }
    }

    fn solve(&mut self) -> u8 {
        let _len: u8 = self.scan.token();
        let s: String = self.scan.token();
        let mut max = 1;
        for c in s.chars() {
            let pos = c as u8 - b'a' + 1;
            if pos > max { max = pos; }
        }
        
        max
    }
}



fn main() {
    let mut solution: Solution<StdinLock> = Solution::new(io::stdin().lock());
    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        let ans = solution.solve();
        writeln!(solution.out, "{}", ans);
    }
}

#[test]
fn test_sample() {
    let f = File::open("test_inputs/codeforces/round835_div4/b.in").unwrap();
    let f = BufReader::new(f);
    let mut solution: Solution<BufReader<File>> = Solution::new(f);

    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        let ans = solution.solve();
        writeln!(solution.out, "{}", ans);
    }
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
