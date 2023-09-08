use std::io::{stdout, Write};
use std::fs::File;
use std::time::{Duration, Instant};
use std::thread;
use std::sync::{Arc, Mutex};
use rand::SeedableRng;
use rand::distributions::{Distribution, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;


#[derive(Debug)]
pub(crate) struct AmidakujiResult {
    pub(crate) result: Vec<(usize, u64)>,
    pub(crate) elapsed: Duration
}

impl AmidakujiResult {
    pub(crate) fn fmt_elapsed (&self) -> String {
        let elapsed = self.elapsed;
        format!("{:>02}:{:>02}:{:>02}.{:>03}", elapsed.as_secs() / 3600, elapsed.as_secs() / 60 % 60, elapsed.as_secs() % 60, elapsed.subsec_millis())
    }
    
    pub(crate) fn output (&self, path: &str) {
        let result_string = self.result.iter().fold("x, y\n".to_string(), |str, (v, h)| {
            str + &format!("{}, {}\n", v, h)
        });
        let mut file = File::create(path).unwrap();
        write!(file, "{}", result_string).unwrap();
        file.flush().unwrap();
        
    }
}


fn check_fair (goals: Vec<usize>, vertical: usize, count: usize) -> bool {
    let vertical = vertical as f64;
    let count = count as f64;
    goals.iter().all(|&x| count * 0.95 / vertical <= x as f64 && x as f64 <= count * 1.05 / vertical)
}


pub(crate) fn normal_amidakuji (vertical_range: impl Iterator<Item = usize> + Clone, count: usize, accuracy: usize, acc_base: f64, thread_count: usize) -> AmidakujiResult {
    let time = Instant::now();
    let range_length = vertical_range.clone().count();
    let digits = range_length.to_string().len();
    print!("{} / {} [{}] 00:00:00", "0".repeat(digits), range_length, "-".repeat(accuracy));
    stdout().flush().unwrap();
    let finished = Arc::new(Mutex::new(false));
    let handler =
        {
            let finished = finished.clone();
            thread::spawn(move || {
                while !(*finished.lock().unwrap()) {
                    thread::sleep(Duration::from_secs(1));
                    let elapsed = time.elapsed().as_secs();
                    print!("\x1b[8D{:>02}:{:>02}:{:>02}", elapsed / 3600, elapsed / 60 % 60, elapsed % 60);
                    stdout().flush().unwrap();
                }
            })
        };
    let result =
        vertical_range.enumerate().map(|(nth, vertical)| {
            let mut needed_horizontal = 0;
    
            let mut last_index = 0.;
            let mut index = 1.;
            for i in 0..accuracy {
                let horizontal = acc_base.powf(index).floor() as u64;
    
                let mut handlers = vec![];
                for _ in 0..thread_count {
                    handlers.push(thread::spawn(move || {
                        let mut rng = Xoshiro256PlusPlus::from_entropy();
                        let range = Uniform::new(0, vertical-1);
                        let mut goals = vec![0_usize; vertical];
                        // When count % thread_count != 0, total trial != count
                        for _ in 0..count/thread_count {
                            let mut current_pos = 0;
                            for _ in 0..horizontal {
                                let line = range.sample(&mut rng);
                                if current_pos == line {
                                    current_pos += 1;
                                } else if current_pos == line + 1 {
                                    current_pos -= 1;
                                }
                            }
                            goals[current_pos] += 1;
                        }
                        goals
                    }));
                }
                
                let goals = handlers.into_iter().fold(vec![0; vertical], |prev, x| {
                    x.join().unwrap().iter().zip(prev).map(|(a, b)| a + b).collect()
                });
    
                if check_fair(goals, vertical, count) {
                    needed_horizontal = horizontal;
                    index = (last_index + index) / 2.;
                } else if index % 1. == 0. {
                    (index, last_index) = (index + 1., index);
                } else {
                    (index, last_index) = (index + (index - last_index) / 2., index);
                }
                print!("\r{:>0digits$} / {} [\x1b[32m{}\x1b[36m{}\x1b[0m]\x1b[9C", nth+1, range_length, "#".repeat(i + 1), "-".repeat(accuracy - i - 1), digits = digits);
                stdout().flush().unwrap();
            }
            (vertical, needed_horizontal)
        }).collect();
    *finished.lock().unwrap() = true;
    handler.join().unwrap();
    let result = AmidakujiResult {
        result,
        elapsed: time.elapsed()
    };
    println!("\r\x1b[2K{0} / {0} [\x1b[32mFinished\x1b[0m] {1}", range_length, result.fmt_elapsed());
    result
}
