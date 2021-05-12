import sys
from lgn.models.autotest.lgn_tests import lgn_tests
import torch
import os.path as osp

import numpy as np
from math import sqrt, cosh
import matplotlib.pyplot as plt

from lgn.models.lgn_encoder import LGNEncoder
from lgn.models.lgn_decoder import LGNDecoder

from args import setup_argparse

from utils.make_data import initialize_data
from utils.utils import save_data


args = setup_argparse()

encoder = LGNEncoder(num_input_particles=args.num_jet_particles,
                     tau_input_scalars=args.tau_jet_scalars,
                     tau_input_vectors=args.tau_jet_vectors,
                     map_to_latent=args.map_to_latent,
                     tau_latent_scalars=args.tau_latent_scalars,
                     tau_latent_vectors=args.tau_latent_vectors,
                     maxdim=args.maxdim, max_zf=[1],
                     num_channels=args.encoder_num_channels,
                     weight_init=args.weight_init, level_gain=args.level_gain,
                     num_basis_fn=args.num_basis_fn, activation=args.activation, scale=args.scale,
                     mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                     device=args.device, dtype=args.dtype)
decoder = LGNDecoder(tau_latent_scalars=args.tau_latent_scalars,
                     tau_latent_vectors=args.tau_latent_vectors,
                     num_output_particles=args.num_jet_particles,
                     tau_output_scalars=args.tau_jet_scalars,
                     tau_output_vectors=args.tau_jet_vectors,
                     maxdim=args.maxdim, max_zf=[1],
                     num_channels=args.decoder_num_channels,
                     weight_init=args.weight_init, level_gain=args.level_gain,
                     num_basis_fn=args.num_basis_fn, activation=args.activation,
                     mlp=args.mlp, mlp_depth=args.mlp_depth, mlp_width=args.mlp_width,
                     device=args.device, dtype=args.dtype)

path = './autoencoder_trained_models/LGNAutoencoder_gJet_sum_tauLS3_tauLV2_encoderCG2321_decoderCG2321'
encoder_path = osp.join(path, 'weights_encoder')
decoder_path = osp.join(path, 'weights_decoder')

encoder.load_state_dict(torch.load(f'{encoder_path}/epoch_1_encoder_weights.pth', map_location='cpu'))
decoder.load_state_dict(torch.load(f'{decoder_path}/epoch_1_decoder_weights.pth', map_location='cpu'))

train_loader, test_loader, valid_loader = initialize_data(path='./hls4ml/cartesian/g_jets_150p_cartesian.pt',
                                                          batch_size=5,
                                                          num_train=2,
                                                          num_test=2,
                                                          num_val=2)

beta_max=10.0
dev = lgn_tests(encoder, decoder, test_loader, args, beta_max=beta_max, epoch=args.num_epochs, cg_dict=encoder.cg_dict)
save_data(dev, f'equivariance_test_results_beta_max_{beta_max}', is_train=None, outpath='./', epoch=-1)


dev_boost_p4 = []
dev_boost_scalars = []
dev_boost = dev['boost_dev_output']
for i in range(len(dev_boost)):
    dev_boost_p4.append(dev_boost[i][(1,1)].item())
    dev_boost_scalars.append(dev_boost[i][(0,0)].item())

gammas = dev['gammas']

plt.plot(gammas, dev_boost_p4, label='four-momenta')
plt.plot(gammas, dev_boost_scalars, label='scalars')
plt.legend()
plt.title('Boost equivariance test')
plt.xlabel(r'Boost factor $\gamma$')
plt.ylabel('Relative deviation')
plt.savefig('boost.pdf')
plt.savefig('boost.png', dpi=600)
plt.close()

dev_rot_p4 = []
dev_rot_scalars = []
dev_rot = dev['rot_dev_output']
for i in range(len(dev_rot)):
    dev_rot_p4.append(dev_boost[i][(1,1)].item())
    dev_rot_scalars.append(dev_rot[i][(0,0)].item())

thetas = dev['thetas']

plt.plot(thetas, dev_rot_p4, label='four-momenta')
plt.plot(thetas, dev_rot_scalars, label='scalars')
plt.legend()
plt.title('Rotation equivariance test')
plt.xlabel(r'Rotation angle $\theta$ (rad)')
plt.ylabel('Relative deviation')
plt.savefig('rotation.pdf')
plt.savefig('rotation.png', dpi=600)
plt.close()
