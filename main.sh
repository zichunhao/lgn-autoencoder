#!/bin/bash

if [ "$#" -eq 1 ]; then
    output_path=$1
    num_epochs=10
    train_fraction=10
    num_valid=100
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
    output_path="./autoencoder-trained-models-test"
    num_epochs=10
    train_fraction=20
    num_valid=20
fi

mkdir -p "$output_path"/autoencoder;
python main.py \
-b 16 \
-j g \
-e 10 \
--train-fraction "$train_fraction" \
--num-valid "$num_valid" \
--maxdim 2 \
--unit TeV \
--loss-choice mse \
--loss-norm-choice p3 \
--im False \
--tau-latent-scalars 1 \
--tau-latent-vectors 1 \
--file-suffix jets_30p_p4 \
--num-test-batch 200 \
--equivariance-test \
--encoder-num-channels 4 5 4 4 \
--decoder-num-channels 4 5 4 4 \
--file-path "./hls4ml" \
--file-suffix "jets_30p_p4" \
--save-dir "$output_path" \
| tee -a "$output_path"/autoencoder-g-s1-v1-4544-4544.txt
