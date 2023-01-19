// tourist solution
// https://codeforces.com/contest/1779/submission/187822306

use std::{
    collections::HashSet,
        fs::File,
        io::{self, BufRead, BufReader, BufWriter, StdinLock, StdoutLock, Write},
        ops::{Add, Sub},
        str::{self, FromStr},
};
use std::slice::Iter;

fn upper_bound(v: &[i64], val: i64) -> Iter<'_, i64> {
    match v.iter().position(|n| *n > val) {
        Some(pos) => v[pos..].iter(),
        None => v[v.len()..].iter(),
    }
}

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

    fn win(mut a: Vec<i64>, k: usize, s: i64) -> bool {
        let n = a.len();
        if n == k {
            let opp = a.iter().sum();
            return s >= opp;
        }
        a.sort_by(|a, b| b.cmp(a));
        if k == 1 {
            for i in 0..n {
                if s >= a[i] {
                    // let mut b = a.clone();
                    let ai = a[i];
                    a.remove(i);
                    return Self::win(a, 2, s + ai);
                }
            }
            return false;
        }
        if k == 2 {
            let mut mn = n;
            for i in 0..n {
                let mut val = n;
                for j in i+1..n {
                    if s >= a[i] + a[j] {
                        val = j;
                        break;
                    }
                }
                if val < mn {
                    mn = val;
                    let mut b = a.clone();
                    b.remove(val);
                    b.remove(i);
                    if Self::win(b, 4, s + a[i] + a[val]) {
                        return true;
                    }
                }
            }
            return false;
        }
        if k == 4 {
            let mut mn: Vec<Vec<Vec<usize>>> = vec![vec![vec![n; n]; n]; n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        let mut val = n;
                        if i < j && j < k {
                            for t in k+1..n {
                                if s >= a[i] + a[j] + a[k] + a[t] {
                                    val = t;
                                    break;
                                }
                            }
                        }
                        mn[i][j][k] = n;
                        if i > 0  { mn[i][j][k] = mn[i][j][k].min(mn[i - 1][j][k]); }
                        if j > 0  { mn[i][j][k] = mn[i][j][k].min(mn[i][j - 1][k]); }
                        if k > 0  { mn[i][j][k] = mn[i][j][k].min(mn[i][j][k - 1]); }
                        if val < mn[i][j][k] {
                            mn[i][j][k] = val;
                            let mut b = a.clone();
                            b.remove(val);
                            b.remove(k);
                            b.remove(j);
                            b.remove(i);
                            if Self::win(b, 8, s + a[i] + a[j] + a[k] + a[val]) {
                                return true;
                            }
                        }
                    }
                }
            }
            return false;
        }
        if k == 8 {
            assert_eq!(24, n);
            let total: i64 = a.iter().sum::<i64>() + s;
            let goal = (total + 1) / 2;
            if 2 * s < goal {
                return false;
            }
            let mut all: Vec<Vec<i64>> = vec![vec![]; 9];
            fn dfs(all: &mut [Vec<i64>], a: &[i64], v: usize, w: usize, sum: i64, s: i64, goal: i64) -> bool {
                if v == 12 {
                    if sum <= s {
                        if s + sum >= goal {
                            return true;
                        } else {
                            all[w].push(sum);
                        }
                    }
                    return false;
                }
                if w < 8 {
                    if dfs(all, a, v + 1, w + 1, sum + a[v], s, goal) {
                        return true;
                    }
                }
                dfs(all, a, v + 1, w, sum, s, goal)
            }
            let found = dfs(&mut all, &a, 0, 0, 0, s, goal);
            if found {
                return true;
            }
            for i in 0..=8 {
                all[i].sort();
            }

            return Self::find(&all, &a, 12, 0, 0, s, goal);
        }
        unreachable!();
    }

    fn find(all: &[Vec<i64>], a: &[i64], v: usize, w: usize, mut sum: i64, s: i64, goal: i64) -> bool {
        if v == 24 {
            let vec = &all[8 - w];
            match vec.iter().position(|n| *n > s - sum) {
                Some(pos) => {
                    if pos > 0 {
                        sum += vec[pos - 1];
                    }
                }
                None => {
                    if vec.len() > 0 {
                        sum += vec[vec.len() - 1];
                    }
                }
            }
            if s + sum >= goal {
                return true;
            }
            return false;
        }
        if w < 8 {
            if Self::find(all, a, v + 1, w + 1, sum + a[v], s, goal) {
                return true;
            }
        }
        Self::find(all, a, v + 1, w, sum, s, goal)
    }

    fn solve(&mut self) {
        let n: usize = self.scan.token();
        let a: Vec<i64> = self.scan.read_nums();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|i, j| a[*i].cmp(&a[*j]));
        let mut low: i32 = -1;
        let mut high: i32 = n as i32 - 1;
        while low + 1 < high {
            let mid: i32 = (low + high) >> 1;
            let mut b = a.clone();
            b.remove(order[mid as usize]);
            if Self::win(b, 1, a[order[mid as usize]]) {
                high = mid;
            } else {
                low = mid;
            }
        }
        let mut res: Vec<char> = vec!['0'; n];
        for i in high as usize..n {
            res[order[i]] = '1';
        }

        writeln!(self.out, "{}", res.into_iter().collect::<String>());
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
fn test_interactive() {
    main();
}


#[test]
fn test_sample() {
    let fr = File::open("tests_in_out/codeforces/hello2023/h.in").unwrap();
    let fr = BufReader::new(fr);
    let out_file = File::create("tests_in_out/codeforces/hello2023/h.out").unwrap();
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
