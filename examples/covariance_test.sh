#!/bin/bash

python covariance_test.py \
-tbs 4 \
--num-test-batch 1 \
-j g \
--maxdim 2 \
--tau-latent-scalars 1 \
--tau-latent-vectors 13 \
--file-suffix jets_30p_p4 \
--encoder-num-channels 4 5 3 3 \
--decoder-num-channels 4 5 3 3 \
--model-path 'autoencoder-trained-models-test/LGNAutoencoder_gJet_mean_tauLS1_tauLV13_encoder4533_decoder4533' \
--data-path "./data/g_jets_30p_p4.pt" \
--test-data-path "./data/g_jets_30p_p4_test.pt" \
| tee -a 'autoencoder-trained-models-test/autoencoder-g-s1-v1-4533-4533-covariance-test.txt'
