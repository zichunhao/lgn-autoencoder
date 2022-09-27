#!/bin/bash

set -xe

python test.py \
-tbs 500 \
--num-test-batch 200 \
-j QCD \
--maxdim 2 \
--tau-latent-vectors 8 \
--tau-latent-scalars 1 \
--loss-choice chamfer \
--get-real-method sum \
--map-to-latent "min&max" \
--mlp-width 6 \
--mlp-depth 6 \
--encoder-num-channels 3 3 4 4 \
--decoder-num-channels 4 4 3 3 \
--model-path "dev/exp/LGNAutoencoder_QCDJet_min&max_tauLS1_tauLV8_encoder3344_decoder4433" \
--anomaly-detection \
--data-paths "data/g_jets_30p_p4_small.pt" "data/q_jets_30p_p4_small.pt" \
--test-data-paths "data/g_jets_30p_p4_small.pt" "data/q_jets_30p_p4_small.pt" \
--anomaly-detection \
--signal-paths "data/t_jets_30p_p4_small.pt" "data/w_jets_30p_p4_small.pt" "data/z_jets_30p_p4_small.pt" \
--signal-types t w z \
--equivariance-test \
| tee -a "dev/exp/test-log.txt"
