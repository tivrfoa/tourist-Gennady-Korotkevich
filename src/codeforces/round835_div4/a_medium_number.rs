// https://codeforces.com/contest/1760/submission/181893774

use io::Write;
use std::{io, str};

fn swap(a: &mut u8, b: &mut u8) {
	let tmp = *a;
	*a = *b;
	*b = tmp;
}

fn main() {
    let (stdin, stdout) = (io::stdin(), io::stdout());
    let mut scan = UnsafeScanner::new(stdin.lock());
    let mut out = io::BufWriter::new(stdout.lock());
    let t = scan.token::<usize>();

	for _ in 0..t {
		let (mut a, mut b, mut c) = get_3_nums::<u8>(&mut scan);
		if a > b { swap(&mut a, &mut b); }
		if a > c { swap(&mut a, &mut c); }
		if b > c { swap(&mut b, &mut c); }
        writeln!(out, "{}", b);
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
