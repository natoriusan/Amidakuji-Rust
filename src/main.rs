use crate::functions::*;

mod functions;

fn main() {
    let results = construct_connected_amidakuji().calculate(3..=100, Some(100));
    // let results = construct_connected_amidakuji().calculate_parallel(3..=100, 30, Some(100));
    results.output_to_file("output/amidakuji.csv");
}
