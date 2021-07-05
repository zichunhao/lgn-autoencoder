#!/bin/bash

mkdir -p ./autoencoder-trained-models/autoencoder;
python main.py \
-j g \
-e 1000 \
-bs 110 \
--train-fraction 0.75 \
--lr 1e-4 \
--save-freq 1000 \
--tau-latent-scalars 1 \
--tau-latent-vectors 1 \
--equivariance-test \
--num-test-batch 200 \
--encoder-num-channels 4 5 4 4 \
--decoder-num-channels 4 5 4 4 \
--file-path "./hls4ml" \
--file-suffix jets_30p_p4 \
--save-dir "./autoencoder-trained-models/autoencoder" \
| tee -a "./autoencoder-trained-models/autoencoder-g-s1-v1-4544-4544.txt"
