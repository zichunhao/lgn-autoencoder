#!/bin/bash

mkdir -p ./autoencoder-trained-models/autoencoder;
python main.py \
-b 2 \
-j g \
-e 5 \
--train-fraction 2 \
--num-valid 2 \
--save-dir autoencoder-trained-models-test \
| tee -a "./autoencoder-trained-models-test/autoencoder-g-s1-v1-4544-4544.txt"
