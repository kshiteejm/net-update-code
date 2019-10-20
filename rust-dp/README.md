Compile code (remove `--release` for faster compilation)
```
cargo build --release
```

Run compiled code with an example (replace `/release/` with `/debug/` when `--release` flag is off)
```
./target/release/rust-dp --num-nodes 4 --num-steps 3 --update-idx 1 2 3 --cm-path ./data/cm.csv
```

The result by default is written in
```
data/result.csv
```
