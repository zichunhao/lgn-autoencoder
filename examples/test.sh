#!/bin/bash

python test.py \
-tbs 4 \
--num-test-batch 200 \
-j g \
--maxdim 2 \
--unit "TeV" \
--tau-latent-scalars 1 \
--tau-latent-vectors 1 \
--file-suffix jets_30p_p4 \
--encoder-num-channels 4 5 4 4 \
--decoder-num-channels 4 5 4 4 \
--model-path 'autoencoder-trained-models-test/LGNAutoencoder_gJet_mean_tauLS1_tauLV1_encoder4544_decoder4544' \
--file-path "hls4ml" \
--file-suffix "jets_30p_p4" \
--equivariance-test \
| tee -a 'autoencoder-trained-models-test/autoencoder-g-s1-v1-4544-4544-covariance-test.txt'
