#!/bin/bash

output_path="./autoencoder-trained-models-test"

mkdir -p "$output_path"/autoencoder;
python main.py \
-b 2 \
-j g \
-e 10 \
--train-fraction 10 \
--num-valid 10 \
--maxdim 2 \
--unit TeV \
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
