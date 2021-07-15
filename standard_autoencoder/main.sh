#!/bin/bash

mkdir -p ./standard-autoencoder-trained-models/autoencoder;
python main.py \
-b 2 \
-j g \
-e 5 \
--train-fraction 2 \
--num-valid 2 \
--save-dir standard-autoencoder-trained-models \
| tee -a "./standard-autoencoder-trained-models/autoencoder-g-s1-v1-4544-4544.txt"
