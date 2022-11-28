#!/bin/bash

export IFS=","
cat all_ttfs.csv | while read x id name; 
do 
echo "$id:$name"; 
python ttf_optimizer.py -i $id -d 20221124 -o all_ttf_weights_20221124.csv
done
