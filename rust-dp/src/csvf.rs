extern crate csv;
extern crate serde;

use serde::Deserialize;
use serde::Serialize;
use std::collections::HashMap;
use std::error::Error;


#[derive(Deserialize)]
struct InputRecord {
    cost: f32,
    down_idx: Vec<u32>,
}


#[derive(Serialize)]
struct ValueRecord {
    value: f32,
    step: u32,
    update_idx: Vec<u32>,
}


#[derive(Serialize)]
struct ActionRecord {
    action: Vec<u32>,
    divider: String,
    step: u32,
    update_idx: Vec<u32>,
}


pub fn read_csv_file(file_name: String)-> Result<
                     HashMap<Vec<u32>, f32>,
                     Box<dyn Error>> {

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b',')
        .flexible(true)
        .from_path(file_name)?;

    let mut results: HashMap<Vec<u32>, f32> = 
        HashMap::new();

    for result in rdr.deserialize() {
        let record: InputRecord = result?;
        results.insert(record.down_idx, record.cost);
    }

    Ok(results)
}


pub fn write_value_csv(file_name: String,
                      values: &HashMap<(u32, Vec<u32>), f32>)
                      -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b',')
        .flexible(true)
        .has_headers(false)
        .from_path(file_name)?;

    for (key, val) in values.iter() {
        wtr.serialize(ValueRecord {
            value: *val,
            step: key.0,
            update_idx: key.1.to_vec(),
        })?;
    }
    Ok(())
}


pub fn write_action_csv(file_name: String,
                      actions: &HashMap<(u32, Vec<u32>), Vec<u32> >)
                      -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b',')
        .flexible(true)
        .has_headers(false)
        .from_path(file_name)?;

    for (key, val) in actions.iter() {
        wtr.serialize(ActionRecord {
            action: val.to_vec(),
            divider: '*'.to_string(),
            step: key.0,
            update_idx: key.1.to_vec(),
        })?;
    }
    Ok(())
}

