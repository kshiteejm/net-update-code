mod conf;
mod csvf;
mod dp;

extern crate itertools;

use structopt::StructOpt;


fn main() {
    // Parse command line arguments
    let config = conf::Config::from_args();

    // Read csv file in format:
    // cost, down switch indices
    println!("Reading csv file");
    let mut cost_model = match csvf::read_csv_file(
                               config.cm_path) {
        Ok(results) => results,
        Err(err) => panic!("csv read file error: {}", err),
    };

    // Add dummy no switch down value to cost model
    cost_model.insert([].to_vec(), 0.0);

    // Run DP
    let results = dp::dp(config.num_nodes,
                         config.num_steps,
                         &config.update_idx,
                         &cost_model);

    // Write results into a csv file in format:
    // value, step, index to update (flexible len)
    match csvf::write_csv_file(config.result_path,
                         &results) {
        Ok(()) => println!("Results written in file."),
        Err(err) => panic!("csv write file error: {}", err),
    };
}
