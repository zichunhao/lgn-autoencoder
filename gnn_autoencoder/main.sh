#!/bin/bash

if [ "$#" -eq 1 ]; then
    output_path=$1
    num_epochs=10
    train_fraction=10
    num_valid=10
elif [[ "$#" -eq 2 ]]; then
    output_path=$1
    num_epochs=$2
    train_fraction=10
    num_valid=10
elif [[ "$#" -eq 4 ]]; then
    output_path=$1
    num_epochs=$2
    train_fraction=$3
    num_valid=$4
else
    output_path="./standard-autoencoder-trained-models"
    num_epochs=10
    train_fraction=10
    num_valid=10
fi

mkdir -p "$output_path";
python main.py \
-b 2 \
-j g \
-e $num_epochs \
--latent-map 'local' \
--latent-node-size 1 \
--encoder-edge-sizes '16,16,8,8;' \
--decoder-edge-sizes '16,16,8,8;' \
--encoder-node-sizes '3;3;3;3;' \
--decoder-node-sizes '3;3;3;3;' \
--chamfer-jet-features-weight 10 \
--train-fraction "$train_fraction" \
--num-valid "$num_valid" \
--save-dir "$output_path" \
| tee -a "$output_path"/autoencoder-g-s1-v1-4544-4544.txt
