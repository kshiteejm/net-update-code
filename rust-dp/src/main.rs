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
    let (values, actions) = dp::dp(config.num_nodes,
                            config.num_steps,
                            &config.update_idx,
                            &cost_model);

    // Write values into a csv file in format:
    // value, step, index to update (flexible len)
    match csvf::write_value_csv(config.value_path,
                         &values) {
        Ok(()) => println!("Values written in file."),
        Err(err) => panic!("csv write file error: {}", err),
    };
    // Write values into a csv file in format:
    // next index to update, *, step, index to update (flexible len)
    // '*' is for knowing the breaking point
    match csvf::write_action_csv(config.action_path,
                         &actions) {
        Ok(()) => println!("Actions written in file."),
        Err(err) => panic!("csv write file error: {}", err),
    };
}
