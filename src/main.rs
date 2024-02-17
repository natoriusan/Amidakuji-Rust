use crate::functions::*;

mod functions;

fn main() {
    let results = construct_connected_amidakuji().calculate_parallel(3..=60, 30);
    results.output_to_file("output/amidakuji.csv");
}
