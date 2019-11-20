#!/bin/bash

START=$1
END=$2

for s in $(seq $START $END)
do
cd environment
python3 fat_tree_network.py $s True True True
cd -
done

