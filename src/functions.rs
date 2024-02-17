#![allow(dead_code)]

use std::cmp::Ordering;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::collections::VecDeque;
use std::fs::File;
use std::io::{stdout, Write};
use std::ops::{AddAssign, MulAssign, ShlAssign};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use itertools::Itertools;


#[derive(Clone, Debug)] 
pub(crate) struct BaseN {
    mantissa: VecDeque<usize>,
    exponent: usize,
    base: usize,
    accuracy: usize,
    max_error: usize
}

impl BaseN {
    fn new(base: usize, accuracy: usize, value: usize) -> Self {
        let mut num =
            Self {
                mantissa: VecDeque::from([value]),
                exponent: 0,
                base,
                accuracy,
                max_error: 0
            };
        num.fix_digits();
        num.round_down();
        num
    }

    fn adjust_exponent(&mut self, exponent: usize) {
        for _ in 0..self.exponent.saturating_sub(exponent) {
            self.mantissa.push_front(0);
        }
        self.exponent = self.exponent.min(exponent);
    }

    fn fix_digits(&mut self) {
        fix_digits(&mut self.mantissa, self.base);
        // let mut i = 0;
        // while i < self.mantissa.len() {
        //     if self.mantissa[i] >= self.base {
        //         if i == self.mantissa.len() - 1 {
        //             self.mantissa.push_back(0);
        //         }
        //         self.mantissa[i+1] += self.mantissa[i] / self.base;
        //         self.mantissa[i] %= self.base;
        //     }
        //     i += 1;
        // }
    }

    fn round_down(&mut self) {
        self.exponent += self.mantissa.len().saturating_sub(self.accuracy);
        for _ in 0..self.mantissa.len().saturating_sub(self.accuracy) {
            self.mantissa.pop_front();
        }
    }
    
    fn get_error_max(&self) -> Self {
        let mut num = self.clone();
        while num.mantissa.len() < self.max_error + 1 {
            num.mantissa.push_back(0);
        }
        num.mantissa[self.max_error] += 1;
        num.fix_digits();
        num
    }
}

impl ShlAssign<usize> for BaseN {
    fn shl_assign(&mut self, rhs: usize) {
        self.exponent += rhs;
    }
}

impl AddAssign<&BaseN> for BaseN {
    fn add_assign(&mut self, rhs: &BaseN) {
        debug_assert_eq!(self.base, rhs.base);
        debug_assert_eq!(self.accuracy, rhs.accuracy);
        let lhs_length = self.mantissa.len();
        let lhs_exponent = self.exponent;
        self.adjust_exponent(rhs.exponent);
        for (i, x) in rhs.mantissa.iter().enumerate() {
            while i + (rhs.exponent - self.exponent) >= self.mantissa.len() {
                self.mantissa.push_back(0);
            }
            self.mantissa[i + (rhs.exponent - self.exponent)] += x;
        }
        self.fix_digits();
        self.max_error = self.max_error.max(rhs.max_error);
        if !(self.mantissa.len() <= self.accuracy && (self.exponent == 0 || rhs.exponent == 0)) {
            self.max_error = self.max_error.max(
                (self.mantissa.len() + self.exponent).abs_diff(
                    rhs.mantissa.len() + rhs.exponent
                )
            );
            self.max_error = self.max_error.max(
                (self.mantissa.len() + self.exponent).abs_diff(
                    lhs_length + lhs_exponent
                )
            );
        }
        self.round_down()
    }
}

impl MulAssign<usize> for BaseN {
    fn mul_assign(&mut self, rhs: usize) {
        self.mantissa.iter_mut().for_each(|x| *x *= rhs);
        self.fix_digits();
    }
}

impl PartialEq for BaseN {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Equal
    }
}

impl Eq for BaseN {

}

impl PartialOrd for BaseN {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BaseN {
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert_eq!(self.base, other.base);
        debug_assert_eq!(self.accuracy, other.accuracy);

        match (self.mantissa.len() + self.exponent).cmp(&(other.mantissa.len() + other.exponent)) {
            Equal => {
                self.mantissa.iter().rev()
                    .zip_longest(other.mantissa.iter().rev())
                    .map(|x| (x.clone().left().unwrap_or(&0), x.right().unwrap_or(&0)))
                    .find_map(|(x, y)| {
                        match x.cmp(y) {
                            Equal => None,
                            ord => Some(ord)
                        }
                    })
                    .unwrap_or(Equal)
            }
            ord => ord
        }
    }
}


#[derive(Debug)]
struct OnlyShift {
    mantissa: VecDeque<usize>,
    exponent: usize,
    base: usize
}

impl OnlyShift {
    fn new(base: usize, value: usize) -> Self {
        let mut num =
            Self {
                mantissa: VecDeque::from([value]),
                exponent: 0,
                base
            };
        num.fix_digits();
        num
    }

    fn fix_digits(&mut self) {
        fix_digits(&mut self.mantissa, self.base);
    }
}

impl ShlAssign<usize> for OnlyShift {
    fn shl_assign(&mut self, rhs: usize) {
        self.exponent += rhs;
    }
}

impl PartialEq<OnlyShift> for BaseN {
    fn eq(&self, other: &OnlyShift) -> bool {
        // if self.exponent != 0 || self.mantissa.len() != other.mantissa.len() + other.exponent {
        //     false
        // } else {
        //     // self.exponent == 0 && self.mantissa == other.mantissa
        //     for i in 0..self.mantissa.len() {
        //         if (i < other.exponent && self.mantissa[i] != 0) || (i >= other.exponent && self.mantissa[i] != other.mantissa[i-other.exponent]) {
        //             return false
        //         }
        //     }
        //     true
        // }
        self.partial_cmp(other) == Some(Equal)
    }
}

impl PartialEq<BaseN> for OnlyShift {
    fn eq(&self, other: &BaseN) -> bool {
        other == self
    }
}

impl PartialOrd<OnlyShift> for BaseN {
    fn partial_cmp(&self, other: &OnlyShift) -> Option<Ordering> {
        debug_assert_eq!(self.base, other.base);
        // if self == other {
        //     Some(Equal)
        // } else {

        match (self.mantissa.len() + self.exponent).cmp(&(other.mantissa.len() + other.exponent)) {
            Equal => {
                let ord =
                    self.mantissa.iter().rev()
                        .zip_longest(other.mantissa.iter().rev())
                        .map(|x| (x.clone().left().unwrap_or(&0), x.right().unwrap_or(&0)))
                        .find_map(|(x, y)| {
                            match x.cmp(y) {
                                Equal => None,
                                ord => Some(ord)
                            }
                        });

                match ord {
                    Some(ord) => Some(ord),
                    None => Some(Equal)
                }
            }
            ord => Some(ord)
        }
    }
}

impl PartialOrd<BaseN> for OnlyShift {
    fn partial_cmp(&self, other: &BaseN) -> Option<Ordering> {
        match other.partial_cmp(self) {
            Some(Greater) => Some(Less),
            Some(Less) => Some(Greater),
            ord => ord
        }
    }
}



fn fix_digits(mantissa: &mut VecDeque<usize>, base: usize) {
    let mut i = 0;
    while i < mantissa.len() {
        if mantissa[i] >= base {
            if i == mantissa.len() - 1 {
                mantissa.push_back(0);
            }
            mantissa[i+1] += mantissa[i] / base;
            mantissa[i] %= base;
        }
        i += 1;
    }
}

pub(crate) struct Amidakuji {
    next_fn: fn(&Vec<BaseN>) -> Vec<BaseN>, 
    base_fn: fn(usize) -> usize
}

impl Amidakuji {
    pub(crate) fn new(next_fn: fn(&Vec<BaseN>) -> Vec<BaseN>, base_fn: fn(usize) -> usize) -> Self {
        Self { 
            next_fn,
            base_fn
        }
    }

    pub(crate) fn calculate(&self, vertical_range: impl Iterator<Item=usize> + Clone) -> AmidakujiResult{
        self.calc(vertical_range, false, 1)
    }

    pub(crate) fn calculate_parallel(&self, vertical_range: impl Iterator<Item=usize> + Clone, thread_max: usize) -> AmidakujiResult{
        self.calc(vertical_range, true, thread_max)
    }
    
    fn calc(&self, vertical_range: impl Iterator<Item=usize> + Clone, parallel: bool, thread_max: usize) -> AmidakujiResult {
        let mut results = Vec::new();
        let total_time = Instant::now();
        let mut handlers = Vec::new();
        let counter = Arc::new(Mutex::new(0));
        let counter_max = vertical_range.clone().count();
        let next_fn = self.next_fn;
        let base_fn = self.base_fn;
        let finished = Arc::new(Mutex::new(false));
        let thread_count = Arc::new(Mutex::new(0));
        let timer_handler = {
            let finished = finished.clone();
            let counter = counter.clone();
            thread::spawn(move || {
                while !(*finished.lock().unwrap()) {
                    thread::sleep(Duration::from_secs(1));
                    let counter = counter.lock().unwrap();
                    print!("\r[\x1b[32m{}\x1b[36m{}\x1b[0m] {:>02}:{:>02}", "#".repeat(*counter), "-".repeat(counter_max - *counter), total_time.elapsed().as_secs() / 60, total_time.elapsed().as_secs() % 60);
                    stdout().flush().unwrap();
                }
            })
        };
        for vertical in vertical_range {
            while *thread_count.lock().unwrap() >= thread_max {
                thread::sleep(Duration::from_millis(100));
            }
            let counter = counter.clone();
            *thread_count.lock().unwrap() += 1;
            let thread_count = thread_count.clone();
            handlers.push(thread::spawn(move || {
                let time = Instant::now();
                let base = base_fn(vertical);
                let mut accuracy = 1; // 105f64.log(base as f64).ceil() as usize;
                let result =
                    'outer: loop {
                        let mut current = vec![BaseN::new(base, accuracy, 0); vertical];
                        current[0] = BaseN::new(base, accuracy, 100 * vertical);
                        let mut total_min = OnlyShift::new(base, 95);
                        let mut total_max = OnlyShift::new(base, 105);
                        let mut count = 0;
                        loop {
                            current = next_fn(&current);
                            count += 1;
                            total_min <<= 1;
                            total_max <<= 1;
                            
                            #[inline(always)]
                            fn le_accuracy(lhs: &OnlyShift, rhs: &BaseN) -> Option<bool> {
                                if lhs <= rhs {
                                    Some(true)
                                } else if &rhs.get_error_max() <= lhs {
                                    Some(false)
                                } else {
                                    None
                                }
                            }
                            #[inline(always)]
                            fn accuracy_le(lhs: &BaseN, rhs: &OnlyShift) -> Option<bool> {
                                if &lhs.get_error_max() <= rhs {
                                    Some(true)
                                } else if rhs < lhs {
                                    Some(false)
                                } else {
                                    None
                                }
                            }

                            match (le_accuracy(&total_min, current.iter().min().unwrap()), accuracy_le(current.iter().max().unwrap(), &total_max)) {
                                (Some(true), Some(true)) => break 'outer count,
                                (None, _) | (_, None) => break,
                                _ => (),
                            }
                        }
                        accuracy *= 2;
                    };
                

                let mut counter = counter.lock().unwrap();
                *counter += 1;
                *thread_count.lock().unwrap() -= 1;
                print!("\r[\x1b[32m{}\x1b[36m{}\x1b[0m] {:>02}:{:>02}", "#".repeat(*counter), "-".repeat(counter_max - *counter), total_time.elapsed().as_secs() / 60, total_time.elapsed().as_secs() % 60);
                stdout().flush().unwrap();
                (vertical, result, time.elapsed())
            }));
            if !parallel {
                results.push(handlers.into_iter().last().unwrap().join().unwrap());
                handlers = Vec::new();
            }
        }

        if parallel {
            for handler in handlers {
                results.push(handler.join().unwrap());
            }
        }
        
        *finished.lock().unwrap() = true;
        timer_handler.join().unwrap();

        println!("\r[\x1b[32m{}\x1b[0m] {}s", "#".repeat(counter_max), total_time.elapsed().as_secs_f64());

        AmidakujiResult {
            results,
            total_time: total_time.elapsed()
        }
    }
}

pub(crate) struct  AmidakujiResult {
    results: Vec<(usize, u32, Duration)>,
    total_time: Duration
}

impl AmidakujiResult {
    pub(crate) fn output_to_file(&self, path: &str) {
        let mut file = File::create(path).unwrap();
        let results_string = "x, y\n".to_string() + &self.results.iter().map(|(v, h, _)| format!("{v}, {h}")).join("\n");
        writeln!(file, "{}", results_string).unwrap();
        file.flush().unwrap();
    }
}


pub(crate) fn construct_normal_amidakuji () -> Amidakuji {
    let normal_amidakuji_next_fn =
        |current: &Vec<BaseN>| {
            let vertical = current.len();
            let mut next = current.clone();
            for i in 0..vertical {
                next[i] *= 
                    if i == 0 || i == vertical - 1 {
                        vertical - 2
                    } else {
                        vertical - 3
                    };
            }
            for i in 1..vertical {
                next[i-1] += &current[i];
            }
            for i in 0..vertical-1 {
                next[i+1] += &current[i];
            }
            
            next
        };
    Amidakuji::new(normal_amidakuji_next_fn, |vertical| vertical - 1)
}


pub(crate) fn construct_connected_amidakuji () -> Amidakuji {
    let connected_amidakuji_next_fn =
        |current: &Vec<BaseN>| {
            let vertical = current.len();
            let mut next = current.clone();
            for i in 0..vertical {
                next[i] *= vertical - 2;
            }
            for i in 0..vertical { // modified
                next[(i+vertical-1) % vertical] += &current[i];
            }
            for i in 0..vertical { // modified
                next[(i+1) % vertical] += &current[i];
            }
            
            next
        };
    Amidakuji::new(connected_amidakuji_next_fn, |vertical| vertical)
}


pub(crate) fn construct_connected_amidakuji_with_long_line () -> Amidakuji {
    let connected_amidakuji_with_long_line =
        |current: &Vec<BaseN>| {
            let vertical = current.len();
            let mut next = current.clone();
            for i in 0..vertical {
                next[i] *= vertical * (vertical - 1) / 2 - vertical + 1;
                // Sub実装すれば計算量改善できる
                let mut sum = 
                    if i == 0 { current[1].clone() }
                    else      { current[0].clone() };
                for (j, x) in current.iter().enumerate().skip(if i == 0 { 2 } else { 1 }) {
                    if i != j {
                        sum += x;
                    }
                }
                next[i] += &sum;
            }
            
            next
        };
    Amidakuji::new(connected_amidakuji_with_long_line, |vertical| vertical * (vertical - 1) / 2)
}
