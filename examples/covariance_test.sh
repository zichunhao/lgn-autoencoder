#!/bin/bash

set -xe;

python covariance_test.py \
-j QCD \
--maxdim 2 \
--tau-latent-vectors 8 \
--tau-latent-scalars 1 \
--map-to-latent "min&max" \
--mlp-width 6 \
--mlp-depth 6 \
--encoder-num-channels 3 3 4 4 \
--decoder-num-channels 4 4 3 3 \
--model-path "dev/exp/LGNAutoencoder_QCDJet_min&max_tauLS1_tauLV8_encoder3344_decoder4433" \
--data-paths "data/g_jets_30p_p4_small.pt" "data/q_jets_30p_p4_small.pt" \
--test-data-paths "data/g_jets_30p_p4_small.pt" "data/q_jets_30p_p4_small.pt" \
| tee -a 'dev/exp/covariance-test-log.txt'
