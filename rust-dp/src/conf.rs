use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(name = "config", about = "Parameter configruations")]
pub struct Config {
    /// Number of nodes in the graph
    #[structopt(long, default_value = "4")]
    pub num_nodes: u32,

    /// Number of steps to update
    #[structopt(long, default_value = "1")]
    pub num_steps: u32,

    /// Node indices to update
    #[structopt(long, default_value = "0")]
    pub update_idx: Vec<u32>,

    /// Path to cost model csv file
    #[structopt(long, default_value = "./data/cm.csv")]
    pub cm_path: String,

    /// Path to ouput value result csv file
    #[structopt(long, default_value = "./data/values.csv")]
    pub value_path: String,

    ///Path to ouput action result csv file
    #[structopt(long, default_value = "./data/actions.csv")]
    pub action_path: String
}