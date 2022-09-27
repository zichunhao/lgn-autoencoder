set -xe;

mkdir -p dev/exp;

python -u main.py \
--data-paths "./data/g_jets_30p_p4_small.pt" "./data/q_jets_30p_p4_small.pt" \
--test-data-paths "./data/g_jets_30p_p4_small.pt" "./data/q_jets_30p_p4_small.pt" \
-j QCD \
-e 10 \
-bs 500 \
--train-fraction 0.75 \
--lr 0.0005 \
--loss-choice chamfer \
--get-real-method sum \
--tau-latent-vectors 8 \
--tau-latent-scalars 1 \
--maxdim 2 \
--l1-lambda 1e-8 \
--l2-lambda 0 \
--map-to-latent "min&max" \
--mlp-width 6 \
--mlp-depth 6 \
--encoder-num-channels 3 3 4 4 \
--decoder-num-channels 4 4 3 3 \
--patience 1000 \
--plot-freq 100 \
--save-freq 200 \
--plot-start-epoch 50 \
--equivariance-test \
--num-test-batch 1024 \
--save-dir "dev/exp" \
| tee "dev/exp/training-log.txt"