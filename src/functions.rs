#![allow(dead_code)]

use std::cmp::{max_by, min_by, Ordering};
use std::cmp::Ordering::Equal;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{stdout, Write};
use std::mem::swap;
use std::ops::{Add, Mul};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use itertools::Itertools;
use rug::Float;
use rug::float::Round;
use rug::ops::{AddAssignRound, MulAssignRound};

#[derive(Clone, Debug)]
struct FloatWithExponent {
    value: Float,
    exponent: i64
}

impl FloatWithExponent {
    fn new(mut value: Float) -> Self {
        let exponent = value.get_exp().unwrap_or(0);
        value >>= exponent;
        let exponent = exponent as i64;
        Self {
            value,
            exponent
        }
    }

    fn add_assign_round(&mut self, rhs: &FloatWithExponent, round: Round) {
        debug_assert_eq!(self.value.prec(), rhs.value.prec());
        if self.value.is_zero() {
            *self = rhs.clone();
        } else if !rhs.value.is_zero() {
            let shifted_rhs = rhs.value.clone() >> (self.exponent - rhs.exponent).clamp(-(self.value.prec() as i64), self.value.prec() as i64) as i32;
            self.value.add_assign_round(shifted_rhs, round);
        }
    }

    fn mul_round(mut self, rhs: &FloatWithExponent, round: Round) -> Self {
        self.mul_assign_round(rhs, round);
        self
    }

    fn mul_assign_round(&mut self, rhs: &FloatWithExponent, round: Round) {
        self.value.mul_assign_round(&rhs.value, round);
        self.exponent += rhs.exponent;
    }
}

impl PartialEq for FloatWithExponent {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Equal)
    }
}

impl PartialOrd for FloatWithExponent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.value.is_zero() || other.value.is_zero() || self.exponent == other.exponent {
            self.value.partial_cmp(&other.value)
        } else {
            self.exponent.partial_cmp(&other.exponent)
        }
    }
}


fn checked_float(f: MinMaxBound) -> Option<MinMaxBound> {
    if f.value_min.value.is_infinite() || f.value_min.value.is_nan() || f.value_max.value.is_infinite() || f.value_max.value.is_nan() {
        None
    } else {
        Some(f)
    }
}

fn check_float(f: &MinMaxBound) -> Option<()> {
    if f.value_min.value.is_infinite() || f.value_min.value.is_nan() || f.value_max.value.is_infinite() || f.value_max.value.is_nan() {
        None
    } else {
        Some(())
    }
}


#[derive(Clone, Debug)]
struct MinMaxBound {
    value_min: FloatWithExponent,
    value_max: FloatWithExponent,
}

impl MinMaxBound {
    fn new(value_min: Float, value_max: Float) -> Option<Self> {
        checked_float(
            Self {
                value_min: FloatWithExponent::new(value_min),
                value_max: FloatWithExponent::new(value_max)
            }
        )
    }

    fn identity(precision: u32) -> Option<Self> {
        Self::from_u32(1, precision)
    }

    fn from_u32(value: u32, precision: u32) -> Option<Self> {
        Self::new(
            Float::with_val_round(precision, value, Round::Down).0,
            Float::with_val_round(precision, value, Round::Up).0
        )
    }

    fn sin(k: u32, u: u32, precision: u32) -> Option<Self> {
        // checks if precision is enough to express whole bits of integer
        if k == 0 || precision > k.ilog2() {
            let mut min_f = Float::with_val(precision, k);
            let mut max_f = Float::with_val(precision, k);
            min_f.sin_u_round(u, Round::Down);
            max_f.sin_u_round(u, Round::Up);
            Self::new(min_f, max_f)
        } else {
            None
        }
    }

    fn cos(k: u32, u: u32, precision: u32) -> Option<Self> {
        // check precision to express whole bits of integer
        if k == 0 || precision > k.ilog2() {
            let mut min_f = Float::with_val(precision, k);
            let mut max_f = Float::with_val(precision, k);
            min_f.cos_u_round(u, Round::Down);
            max_f.cos_u_round(u, Round::Up);
            Self::new(min_f, max_f)
        } else {
            None
        }
    }

    fn recip(x: u32, precision: u32) -> Option<Self> {
        Self::new(
            {
                let mut f = Float::with_val_round(precision, x, Round::Up).0;
                f.recip_round(Round::Down);
                f
            },
            {
                let mut f = Float::with_val_round(precision, x, Round::Down).0;
                f.recip_round(Round::Up);
                f
            }
        )
    }

    fn fix_exponent(&mut self) {
        self.value_min.exponent += self.value_min.value.get_exp().unwrap_or(0) as i64;
        self.value_min.value >>= self.value_min.value.get_exp().unwrap_or(0);
        self.value_max.exponent += self.value_max.value.get_exp().unwrap_or(0) as i64;
        self.value_max.value >>= self.value_max.value.get_exp().unwrap_or(0);
    }

    fn add_assign(&mut self, rhs: &MinMaxBound) -> Option<()> {
        self.value_min.add_assign_round(&rhs.value_min, Round::Down);
        self.value_max.add_assign_round(&rhs.value_max, Round::Up);
        
        check_float(self)?;
        self.fix_exponent();
        Some(())
    }

    fn mul_assign(&mut self, rhs: &MinMaxBound) -> Option<()> {
        match ((self.value_max.value.is_sign_positive(), rhs.value_max.value.is_sign_positive()), (self.value_min.value.is_sign_positive(), rhs.value_min.value.is_sign_positive())) {
            ((true, true), (true, true)) => {
                self.value_max.mul_assign_round(&rhs.value_max, Round::Up);
                self.value_min.mul_assign_round(&rhs.value_min, Round::Down);
            },
            ((true, true), (true, false)) => {
                // 順番変えちゃだめ
                self.value_min = self.value_max.clone().mul_round(&rhs.value_min, Round::Down);
                self.value_max.mul_assign_round(&rhs.value_max, Round::Up);
            },
            ((true, true), (false, true)) => {
                self.value_max.mul_assign_round(&rhs.value_max, Round::Up);
                self.value_min.mul_assign_round(&rhs.value_max, Round::Down);
            },
            ((true, true), (false, false)) => {
                self.value_max = max_by(
                    self.value_max.clone().mul_round(&rhs.value_max, Round::Up),
                    self.value_min.clone().mul_round(&rhs.value_min, Round::Up),
                    |x, y| x.partial_cmp(y).unwrap()
                );
                self.value_min = min_by(
                    self.value_max.clone().mul_round(&rhs.value_min, Round::Down),
                    self.value_min.clone().mul_round(&rhs.value_max, Round::Down),
                    |x, y| x.partial_cmp(y).unwrap()
                )
            },
            ((true, false), (true, false)) => {
                self.value_max.mul_assign_round(&rhs.value_min, Round::Down);
                self.value_min.mul_assign_round(&rhs.value_max, Round::Up);
                swap(&mut self.value_max, &mut self.value_min);
            },
            ((true, false), (false, false)) => {
                self.value_max.mul_assign_round(&rhs.value_min, Round::Down);
                self.value_min.mul_assign_round(&rhs.value_max, Round::Up);
                swap(&mut self.value_max, &mut self.value_min);
            },
            ((false, true), (false, true)) => {
                self.value_max.mul_assign_round(&rhs.value_min, Round::Up);
                self.value_min.mul_assign_round(&rhs.value_max, Round::Down);
            },
            ((false, true), (false, false)) => {
                // 順番変えちゃだめ
                self.value_max = self.value_min.clone().mul_round(&rhs.value_min, Round::Up);
                self.value_min.mul_assign_round(&rhs.value_max, Round::Down);
            },
            ((false, false), (false, false)) => {
                self.value_max.mul_assign_round(&rhs.value_max, Round::Down);
                self.value_min.mul_assign_round(&rhs.value_min, Round::Up);
                swap(&mut self.value_max, &mut self.value_min);
            },
            ((false, _), (true, _)) | ((_, false), (_, true)) => unreachable!()
        }


        check_float(self)?;
        self.fix_exponent();
        
        
        Some(())
    }
}

impl Add<&MinMaxBound> for MinMaxBound {
    type Output = Option<Self>;

    fn add(mut self, rhs: &MinMaxBound) -> Self::Output {
        self.add_assign(rhs)?;
        Some(self)
    }
}

impl Mul<&MinMaxBound> for MinMaxBound {
    type Output = Option<Self>;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.mul_assign(rhs)?;
        Some(self)
    }
}

impl Display for MinMaxBound {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{} * 2^{}, {} * 2^{}]", self.value_min.value.clone(), self.value_min.exponent, self.value_max.value.clone(), self.value_max.exponent)
    }
}


#[derive(Clone, Debug)]
struct Vector {
    components: Vec<MinMaxBound>
}


impl Vector {
    fn new(components: impl Into<Vec<MinMaxBound>>) -> Self {
        Self {
            components: components.into()
        }
    }

    fn identity(m: u32, precision: u32) -> Option<Self> {
        Some(
            Self::new(vec![MinMaxBound::identity(precision)?; m as usize])
        )
    }

    fn inner_product(self, rhs: &Vector) -> Option<MinMaxBound> {
        let mut r = (self.components[0].clone() * &rhs.components[0])?;
        check_float(&r)?;
        for (x, y) in self.components.into_iter().zip_eq(&rhs.components).skip(1) {
            r.add_assign(&(x * y)?)?;
        }

        Some(r)
    }

    fn mul_assign(&mut self, rhs: &Vector) -> Option<()> {
        for (x, y) in self.components.iter_mut().zip_eq(&rhs.components) {
            x.mul_assign(y)?;
        }
        Some(())
    }
}

impl Mul<&Vector> for Vector {
    type Output = Option<Self>;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.mul_assign(rhs)?;
        Some(self)
    }
}



pub(crate) struct Amidakuji {
    // m, precision
    diag_fn: fn(u32, u32) -> Option<Vector>,
    // m, from, to, precision
    pre_suf_fn: fn(u32, u32, u32, u32) -> Option<Vector>,
    // m, precision
    total_ratio_fn: fn(u32, u32) -> Option<MinMaxBound>,
    // m -> (from, to)
    least_fn: fn(u32) -> (u32, u32),
    // m -> (from, to)
    most_fn: fn(u32) -> (u32, u32)
}

impl Amidakuji {
    fn new(diag_fn: fn(u32, u32) -> Option<Vector>, pre_suf_fn: fn(u32, u32, u32, u32) -> Option<Vector>, total_ratio_fn: fn(u32, u32) -> Option<MinMaxBound>, least_fn: fn(u32) -> (u32, u32), most_fn: fn(u32) -> (u32, u32)) -> Self {
        Self {
            diag_fn,
            pre_suf_fn,
            total_ratio_fn,
            least_fn,
            most_fn
        }
    }

    pub(crate) fn calculate(&self, vertical_range: impl Iterator<Item=u32> + Clone, max_statusbar_width: Option<u32>) -> AmidakujiResult {
        self.calc(vertical_range, false, 1, max_statusbar_width)
    }

    pub(crate) fn calculate_parallel(&self, vertical_range: impl Iterator<Item=u32> + Clone, thread_max: usize, max_statusbar_width: Option<u32>) -> AmidakujiResult {
        self.calc(vertical_range, true, thread_max, max_statusbar_width)
    }
    
    fn calc(&self, vertical_range: impl Iterator<Item=u32> + Clone, parallel: bool, thread_max: usize, max_statusbar_width: Option<u32>) -> AmidakujiResult {
        let mut results = Vec::new();
        let total_time = Instant::now();
        let mut handlers = Vec::new();
        let counter = Arc::new(Mutex::new(0));
        let vertical_count = vertical_range.clone().count();
        let max_statusbar_width = max_statusbar_width.map(|x| (x as usize).min(vertical_count)).unwrap_or(vertical_count);
        let diag_fn = self.diag_fn;
        let pre_suf_fn = self.pre_suf_fn;
        let total_ratio_fn = self.total_ratio_fn;
        let least_fn = self.least_fn;
        let most_fn = self.most_fn;
        let finished = Arc::new(Mutex::new(false));
        let thread_count = Arc::new(Mutex::new(0));
        let timer_handler = {
            let finished = finished.clone();
            let counter = counter.clone();
            thread::spawn(move || {
                while !(*finished.lock().unwrap()) {
                    thread::sleep(Duration::from_millis(100));
                    let counter = counter.lock().unwrap();
                    print!("\r[\x1b[32m{}\x1b[36m{}\x1b[0m] {:>02}:{:>02}", "#".repeat(*counter * max_statusbar_width / vertical_count), "-".repeat(max_statusbar_width - *counter * max_statusbar_width / vertical_count), total_time.elapsed().as_secs() / 60, total_time.elapsed().as_secs() % 60);
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
                
                let result =
                    (0..)
                        .map(|i| 2_u32.pow(i))
                        .find_map(|precision| {
                            let pre_suf_least = pre_suf_fn(vertical, least_fn(vertical).0, least_fn(vertical).1, precision)?;
                            let pre_suf_most = pre_suf_fn(vertical, most_fn(vertical).0, most_fn(vertical).1, precision)?;
                            let diag = diag_fn(vertical, precision)?;
                            let total_ratio = total_ratio_fn(vertical, precision)?; // MinMaxBound::from_u32(m, precision)?; // MinMaxBound::new(Float::with_val_round(precision, m, Round::Down).0);
                            let low_bound = MinMaxBound::from_u32(95, precision)?;
                            let high_bound = MinMaxBound::from_u32(105, precision)?;
                            let mid_mul = MinMaxBound::from_u32(100 * vertical, precision)?;
                            
                            let check_fairness = |most: MinMaxBound, least: MinMaxBound, total: &MinMaxBound| -> Option<bool> {
                                fold(definitely_le((most * &mid_mul)?, (total.clone() * &high_bound)?), definitely_le((total.clone() * &low_bound)?, (least * &mid_mul)?))
                            };

                            let (diag_powers, total_powers) = {
                                let mut diag_powers = vec![diag];
                                let mut total_powers = vec![total_ratio];

                                for index in 0.. {
                                    if index >= 1 {
                                        diag_powers.push((diag_powers[index-1].clone() * &diag_powers[index-1])?);
                                        total_powers.push((total_powers[index-1].clone() * &total_powers[index-1])?);
                                    }

                                    let most = diag_powers[index].clone().inner_product(&pre_suf_most)?;
                                    let least = diag_powers[index].clone().inner_product(&pre_suf_least)?;
                                    let total = &total_powers[index];
                                    let is_fair = check_fairness(most, least, total)?;
                                    if is_fair {
                                        break
                                    }
                                }

                                (diag_powers, total_powers)
                            };
                            
                            
                            let max_index = diag_powers.len() - 1;
                            {
                                let mut n = 0;
                                let mut last_diag = Vector::identity(vertical, precision)?;
                                let mut last_total = MinMaxBound::identity(precision)?;
                                for index in (0..max_index).rev() {
                                    let next_diag = (last_diag.clone() * &diag_powers[index])?;
                                    let next_total = (last_total.clone() * &total_powers[index])?;

                                    let most = next_diag.clone().inner_product(&pre_suf_most)?;
                                    let least = next_diag.clone().inner_product(&pre_suf_least)?;
                                    let total = &next_total;
                                    
                                    let is_fair = check_fairness(most, least, total)?;

                                    if !is_fair {
                                        n |= 2_u64.pow(index as u32);
                                        last_diag = next_diag;
                                        last_total = next_total;
                                    }
                                }

                                Some(n + 1)
                            }
                        }).unwrap();
                
                let mut counter = counter.lock().unwrap();
                *counter += 1;
                *thread_count.lock().unwrap() -= 1;
                print!("\r[\x1b[32m{}\x1b[36m{}\x1b[0m] {:>02}:{:>02}", "#".repeat(*counter * max_statusbar_width / vertical_count), "-".repeat(max_statusbar_width - *counter * max_statusbar_width / vertical_count), total_time.elapsed().as_secs() / 60, total_time.elapsed().as_secs() % 60);
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

        println!("\r[\x1b[32m{}\x1b[0m] {}s", "#".repeat(max_statusbar_width), total_time.elapsed().as_secs_f64());

        AmidakujiResult {
            results,
            total_time: total_time.elapsed()
        }
    }
}

fn definitely_le(lhs: MinMaxBound, rhs: MinMaxBound) -> Option<bool> {
    if lhs.value_max <= rhs.value_min {
        Some(true)
    } else if rhs.value_max <= lhs.value_min {
        Some(false)
    } else {
        None
    }
}

fn fold(x: Option<bool>, y: Option<bool>) -> Option<bool> {
    Some(x? && y?)
}

pub(crate) struct  AmidakujiResult {
    results: Vec<(u32, u64, Duration)>,
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



fn circulant_pre_suf_fn(vertical: u32, from: u32, to: u32, precision: u32) -> Option<Vector> {
    Some(
        Vector::new(
            (0..vertical).map(|i| {
                let re1 = MinMaxBound::cos(to * i, vertical, precision)?;
                let re2 = MinMaxBound::cos(from * i, vertical, precision)?;

                let im1 = MinMaxBound::sin(to * i, vertical, precision)?;
                let im2 = MinMaxBound::sin(from * i, vertical, precision)?;

                let s = MinMaxBound::recip(vertical, precision)?;

                ((re1 * &re2)? + &(im1 * &im2)?)? * &s
            }).collect::<Option<Vec<_>>>()?
        )
    )
}


pub(crate) fn construct_normal_amidakuji () -> Amidakuji {
    todo!()
}


pub(crate) fn construct_connected_amidakuji () -> Amidakuji {
    let diag_fn =
        |vertical, precision| {
            Some(
                Vector::new(
                    (0..vertical).map(|i| {
                        let f = MinMaxBound::cos(i, vertical, precision)?;
                        let mul = MinMaxBound::from_u32(2, precision)?;
                        let add = MinMaxBound::from_u32(vertical - 2, precision)?; // ::new(Float::with_val_round(precision, m - 2, Round::Down).0);

                        (f * &mul)? + &add
                    }).collect::<Option<Vec<_>>>()?
                )
            )
        };
    
    let total_ratio_fn =
        |vertical, precision| {
            MinMaxBound::from_u32(vertical, precision)
        };
    
    let least_fn =
        |vertical| {
            (0, vertical / 2)
        };
    
    let most_fn = 
        |_| {
            (0, 0)
        };
    
    Amidakuji::new(diag_fn, circulant_pre_suf_fn, total_ratio_fn, least_fn, most_fn)
}

// 左右接続・飛ばすあみだくじについては、必要な横線の式がわかっているため、より高速に計算できる。
pub(crate) fn construct_connected_amidakuji_with_long_line () -> Amidakuji {
    let diag_fn =
        |vertical, precision| {
            Some(
                Vector::new(
                    (0..vertical).map(|i| {
                        if i == 0 {
                            MinMaxBound::from_u32(vertical * (vertical - 1) / 2, precision)
                        } else {
                            MinMaxBound::from_u32(vertical * (vertical - 1) / 2 - vertical, precision)
                        }
                    }).collect::<Option<Vec<_>>>()?
                )
            )
        };

    let total_ratio_fn =
        |vertical, precision| {
            MinMaxBound::from_u32(vertical * (vertical - 1) / 2, precision)
        };

    let least_fn =
        |_| {
            (0, 1)
        };

    let most_fn =
        |_| {
            (0, 0)
        };

    Amidakuji::new(diag_fn, circulant_pre_suf_fn, total_ratio_fn, least_fn, most_fn)
}
