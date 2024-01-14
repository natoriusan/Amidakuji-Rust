use crate::functions::*;

mod functions;

fn main() {
    let results = construct_connected_amidakuji_with_long_line().calculate_parallel(3..=500);
    results.output_to_file("output/amidakuji.csv");
}
