#!/bin/bash

set -xe;

python anomaly_detection.py \
--batch-size 64 \
--bkg-dir "dev/exp/LGNAutoencoder_QCDJet_min&max_tauLS1_tauLV8_encoder3344_decoder4433/test_QCD_jets/anomaly_detection" \
--bkg-type "QCD" \
--sig-dirs \
"dev/exp/LGNAutoencoder_QCDJet_min&max_tauLS1_tauLV8_encoder3344_decoder4433/test_QCD_jets/anomaly_detection/single_signals/t" \
"dev/exp/LGNAutoencoder_QCDJet_min&max_tauLS1_tauLV8_encoder3344_decoder4433/test_QCD_jets/anomaly_detection/single_signals/w" \
"dev/exp/LGNAutoencoder_QCDJet_min&max_tauLS1_tauLV8_encoder3344_decoder4433/test_QCD_jets/anomaly_detection/single_signals/z" \
--sig-types "t" "w" "z" \
| tee -a 'dev/exp/anomaly-detection-log.txt'