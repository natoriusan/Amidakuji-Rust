#![allow(dead_code)]

use std::fs::File;
use std::io::{stdout, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use itertools::Itertools;
use rug::{Complete, Integer};

pub(crate) struct Amidakuji {
    next_fn: fn(&Vec<Integer>) -> (Vec<Integer>, usize)
}

impl Amidakuji {
    pub(crate) fn new(next_fn: fn(&Vec<Integer>) -> (Vec<Integer>, usize)) -> Self {
        Self { next_fn }
    }

    pub(crate) fn calculate(&self, vertical_range: impl Iterator<Item=usize> + Clone) -> AmidakujiResult{
        self.calc(vertical_range, false)
    }

    pub(crate) fn calculate_parallel(&self, vertical_range: impl Iterator<Item=usize> + Clone) -> AmidakujiResult{
        self.calc(vertical_range, true)
    }
    
    fn calc(&self, vertical_range: impl Iterator<Item=usize> + Clone, parallel: bool) -> AmidakujiResult {
        let mut results = Vec::new();
        let total_time = Instant::now();
        let mut handlers = Vec::new();
        let counter = Arc::new(Mutex::new(0));
        let counter_max = vertical_range.clone().count();
        let next_fn = self.next_fn;
        let finished = Arc::new(Mutex::new(false));
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
            let counter = counter.clone();
            handlers.push(thread::spawn(move || {
                let time = Instant::now();
                let mut current = vec![Integer::new(); vertical];
                current[0] = Integer::from(100 * vertical);
                let mut total_min = Integer::from(95);
                let mut total_max = Integer::from(105);
                let mut count = 0;
                loop {
                    let total_change;
                    (current, total_change) = next_fn(&current);
                    count += 1;
                    total_min *= total_change;
                    total_max *= total_change;
                    if &total_min <= current.iter().min().unwrap() && current.iter().max().unwrap() <= &total_max {
                        break;
                    }
                }

                assert_eq!(total_min / 95 * 100 * vertical, current.iter().sum::<Integer>());

                let mut counter = counter.lock().unwrap();
                *counter += 1;
                print!("\r[\x1b[32m{}\x1b[36m{}\x1b[0m] {:>02}:{:>02}", "#".repeat(*counter), "-".repeat(counter_max - *counter), total_time.elapsed().as_secs() / 60, total_time.elapsed().as_secs() % 60);
                stdout().flush().unwrap();
                (vertical, count, time.elapsed())
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
        |current: &Vec<Integer>| {
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
            (next, vertical-1)
        };
    Amidakuji::new(normal_amidakuji_next_fn)
}


pub(crate) fn construct_connected_amidakuji () -> Amidakuji {
    let connected_amidakuji_next_fn =
        |current: &Vec<Integer>| {
            let vertical = current.len();
            let mut next = current.clone();
            for i in 0..vertical {
                next[i] *= vertical - 2;
            }
            // next[1] += &current[0];
            next[vertical-1] += &current[0];
            // next[vertical-2] += &current[vertical-1];
            for i in 1..vertical {
                next[i-1] += &current[i];
            }
            next[0] += &current[vertical-1];
            for i in 0..vertical-1 {
                next[i+1] += &current[i];
            }
            // for i in 1..vertical-1 {
            //     next[i-1] += &current[i];
            //     next[i+1] += &current[i];
            // }
            (next, vertical)
        };
    Amidakuji::new(connected_amidakuji_next_fn)
}



pub(crate) fn construct_connected_amidakuji_with_long_line () -> Amidakuji {
    let connected_amidakuji_with_long_line =
        |current: &Vec<Integer>| {
            let vertical = current.len();
            let mut next = current.clone();
            for i in 0..vertical {
                next[i] *= vertical * (vertical - 1) / 2 - vertical + 1;
            }
            let sum = current.iter().sum::<Integer>();
            for i in 0..vertical {
                next[i] += (&sum - &current[i]).complete();
            }
            (next, vertical * (vertical - 1) / 2)
        };
    Amidakuji::new(connected_amidakuji_with_long_line)
}





// pub(crate) fn normal_amidakuji_2 (vertical_range: impl Iterator<Item = usize> + Clone, accuracy: usize, acc_base: f64) -> AmidakujiResult {
//     fn replicate<T> (n: usize, f: &impl Fn(Vec<T>) -> Vec<T>, x: Vec<T>) -> Vec<T> {
//         // println!("{}", n);
//         // if n == 0 {
//         //     x
//         // } else {
//         //     replicate(n-1, f, f(x))
//         // }
//         let mut v = x;
//         for _ in 0..n {
//             v = f(v);
//         }
//         v
//     }
//     fn next (current: Vec<Integer>) -> Vec<Integer> {
//         // println!("{}", size_of_val(&current));
//         // println!("next started");
//         let r =
//         (0..current.len()).map(|i| {
//             let x = current[i].clone() * (current.len() - 2);
//             if i == 0 {
//                                        x + current[i+1].clone()
//             } else if i == current.len() - 1 {
//                 current[i-1].clone() + x
//             } else {
//                 current[i-1].clone() + x + current[i+1].clone()
//             }
//         }).collect_vec();
//         // println!("next finished");
//         r
//     }
//     let time = Instant::now();
//     let range_length = vertical_range.clone().count();
//     let digits = range_length.to_string().len();
//     print!("{} / {} [{}] 00:00:00", "0".repeat(digits), range_length, "-".repeat(accuracy));
//     stdout().flush().unwrap();
//     let finished = Arc::new(Mutex::new(false));
//     let handler =
//         {
//             let finished = finished.clone();
//             thread::spawn(move || {
//                 while !(*finished.lock().unwrap()) {
//                     thread::sleep(Duration::from_secs(1));
//                     let elapsed = time.elapsed().as_secs();
//                     print!("\x1b[8D{:>02}:{:>02}:{:>02}", elapsed / 3600, elapsed / 60 % 60, elapsed % 60);
//                     stdout().flush().unwrap();
//                 }
//             })
//         };
//     let result =
//         vertical_range.enumerate().map(|(nth, vertical)| {
//             let mut needed_horizontal = 0;
// 
//             let mut last_index = 0.;
//             let mut index = 1.;
//             for i in 0..accuracy {
//                 let horizontal = acc_base.powf(index).floor() as u64;
// 
//                 // let mut handlers = vec![];
//                 // for _ in 0..thread_count {
//                 //     handlers.push(f(vertical, horizontal));
//                 // }
// 
//                 // let goals = handlers.into_iter().fold(vec![0; vertical], |prev, x| {
//                 //     x.join().unwrap().iter().zip(prev).map(|(a, b)| a + b).collect()
//                 // });
//                 
//                 let start = (0..vertical).map(|v| Integer::from(if v == 0 { 1 } else { 0 })).collect_vec();
//                 let goals = replicate(horizontal as usize, &next, start);
//                 
// 
//                 if check_fair_integer(&goals, vertical, &goals.iter().sum()) {
//                     needed_horizontal = horizontal;
//                     index = (last_index + index) / 2.;
//                 } else if index % 1. == 0. {
//                     (index, last_index) = (index + 1., index);
//                 } else {
//                     (index, last_index) = (index + (index - last_index) / 2., index);
//                 }
//                 // println!("check fair");
//                 print!("\r{:>0digits$} / {} [\x1b[32m{}\x1b[36m{}\x1b[0m]\x1b[9C", nth+1, range_length, "#".repeat(i + 1), "-".repeat(accuracy - i - 1), digits = digits);
//                 stdout().flush().unwrap();
//             }
//             (vertical, needed_horizontal)
//         }).collect();
//     *finished.lock().unwrap() = true;
//     handler.join().unwrap();
//     let result = AmidakujiResult {
//         result,
//         elapsed: time.elapsed()
//     };
//     println!("\r\x1b[2K{0} / {0} [\x1b[32mFinished\x1b[0m] {1}", range_length, result.fmt_elapsed());
//     result
// }

