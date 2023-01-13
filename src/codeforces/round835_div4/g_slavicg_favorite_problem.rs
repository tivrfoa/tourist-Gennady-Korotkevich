// tourist solution:
// https://codeforces.com/contest/1760/submission/181923371

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
        let a: usize = self.scan.token::<usize>() - 1;
        let b: usize = self.scan.token::<usize>() - 1;
        let mut g: Vec<Vec<(usize, i32)>> = vec![vec![]; n];

        for _ in 0..n - 1{
            let x: usize = self.scan.token::<usize>() - 1;
            let y: usize = self.scan.token::<usize>() - 1;
            let w: i32 = self.scan.token();
            g[x].push((y, w));
            g[y].push((x, w));
        }

        let d = bfs(&g, a, |u| u != b);
        let all: HashSet<i32> = HashSet::from_iter(d.into_iter().filter(|n| *n != -1));
        let d = bfs(&g, b, |_| true);

        let mut ok = false;
        for i in 0..n {
            if i != b {
                if all.contains(&d[i]) {
                    ok = true;
                    break;
                }
            }
        }

        writeln!(self.out, "{}", if ok { "YES" } else { "NO" });
    }
}

fn bfs<T>(g: &[Vec<(usize, i32)>], from: usize,
        check_from: T) -> Vec<i32>
        where T: Fn(usize) -> bool {
    let mut d: Vec<i32> = vec![-1; g.len()];
    let mut que: Vec<usize> = vec![from];
    d[from] = 0;

    let mut it = 0;
    while it < que.len() {
        for p in &g[que[it]] {
            let u = p.0;
            if d[u] == -1 && check_from(u) {
                que.push(u);
                d[u] = d[que[it]] ^ p.1;
            }
        }

        it += 1;
    }

    d
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
    let fr = File::open("tests_in_out/codeforces/round835_div4/g.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/round835_div4/g.out").unwrap();
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
