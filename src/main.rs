mod functions;

fn main() {
    let results = functions::connected_amidakuji_with_long_horizontal(3..=40, 1000000, 22, 3., 6);
    results.output("output/amidakuji.csv");
}
