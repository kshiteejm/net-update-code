use std::collections::HashMap;


pub fn trace(num_steps: u32,
             update_idx: &Vec<u32>,
             actions: &HashMap<(u32, Vec<u32>), Vec<u32> >)
            -> Vec<Vec<u32> > {

    let mut act_seq: Vec<Vec<u32> > = Vec::new();
    let mut left_idx = update_idx.to_vec();

    for i in 0..num_steps {
        let act = &actions[&(num_steps - i, left_idx.to_vec())];
        act_seq.push(act.to_vec());
        for a in 0..act.len() {
            // This can be O(n^2) but meh (only run once)
            let idx = left_idx.iter().position(
                |x| *x == act[a]).unwrap();
            left_idx.remove(idx);
        }
    }

    // Return results
    act_seq
}