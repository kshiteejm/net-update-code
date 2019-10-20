use std::collections::HashMap;
use std::time::Instant;


fn power_set<T: Clone>(a: &Vec<T>) -> Vec<Vec<T>> {
    a.iter().fold(vec![vec![]], |mut p, x| {
        let i = p.clone().into_iter()
            .map(|mut s| {s.push(x.clone()); s});
        p.extend(i); p})
}


fn exclude_vec(a: Vec<u32>, b: &Vec<u32>) -> Vec<u32> {
    // The input a and b are sorted following same order
    // First vector is copied over and we modify over it
    let mut j = 0;
    let mut c: Vec<u32> = Vec::new();
    for i in 0..a.len() {
        if j >= b.len() {
            c.push(a[i]);
        } else if a[i] == b[j] {
            j += 1;
        } else {
            c.push(a[i]);
        }
    }
    // return vector
    c
}


pub fn dp(num_nodes: u32,
          num_steps: u32,
          update_idx: &Vec<u32>,
          cost_model: &HashMap<Vec<u32>, f32>)
       -> HashMap<(u32, Vec<u32>), f32> {
    
    // Check feasibility
    assert!(num_nodes >= num_steps);
    assert!(num_steps >= update_idx.len() as u32);
    assert!(num_steps >= 1);

    // Check cost_model contains all data
    assert_eq!(cost_model.len() as u32,
               2_u32.pow(update_idx.len() as u32));

    // Initialize data store
    println!("Initialize value hash map");
    let mut values: HashMap<(u32, Vec<u32>), f32> = 
        HashMap::new();
    let uidx_power = power_set(&update_idx);
    for i in 1..=num_steps {
        for j in &uidx_power {
            values.insert((i, j.to_vec()), std::f32::MAX);
        }
    }

    // DP boundary
    // one step left has to take down all switches
    println!("Compute boundary condition (step 1)");
    for j in &uidx_power {
        *values.get_mut(&(1, j.to_vec())).unwrap() = cost_model[j];
    }

    // DP iteration
    for i in 2..=num_steps {
        let now = Instant::now();
        for j in &uidx_power {
            let mut min_v = std::f32::MAX;
            for k in &power_set(&j) {
                let uidx_remain = exclude_vec(j.to_vec(), k);
                let curr_v = values.get(&(i - 1, uidx_remain))
                             .unwrap() + cost_model[k];
                if curr_v < min_v {
                    min_v = curr_v;
                }
            }
            *values.get_mut(&(i, j.to_vec())).unwrap() = min_v;
        }
        println!("DP iteration, step {}, elapsed {} millisecs",
                 i, now.elapsed().as_millis());
    }

    // Return results
    values
}