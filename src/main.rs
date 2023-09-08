mod functions;

fn main() {
    let results = functions::normal_amidakuji(3..=40, 500000, 22, 3., 6);
    results.output("output/amidakuji.csv");
}
