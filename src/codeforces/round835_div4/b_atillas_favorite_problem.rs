use std::{
    fs::File,
    io::{self, BufReader, BufWriter, StdinLock, StdoutLock, BufRead, Write},
    str,
};

struct Solution<R, W: Write> {
    scan: UnsafeScanner<R>,
    out: BufWriter<W>,
}

impl<R: BufRead, W: Write> Solution<R, W> {
    fn new(reader: R, writer: W) -> Self {
        let scan = UnsafeScanner::new(reader);
        let out = BufWriter::new(writer);

        Self { scan, out }
    }

    fn solve(&mut self) -> u8 {
        let _len: u8 = self.scan.token();
        let c: char = self.scan.token::<String>().chars().max().unwrap();
        c as u8 - b'a' + 1
    }
}

fn main() {
    let reader = io::stdin().lock();
    let writer = io::stdout().lock();
    let mut solution: Solution<StdinLock, StdoutLock> = Solution::new(reader, writer);
    let t = solution.scan.token::<usize>();

    for _ in 0..t {
        let ans = solution.solve();
        writeln!(solution.out, "{}", ans);
    }
}

#[test]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/round835_div4/b.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::open("tests_in_out/codeforces/round835_div4/b.out").unwrap();
    let mut solution = Solution::new(fr, out_file);

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
