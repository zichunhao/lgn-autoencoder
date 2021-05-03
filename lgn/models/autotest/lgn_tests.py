import torch
import numpy as np
import time
import numpy.matlib
from math import sqrt, pi, cosh

import logging

from lgn.g_lib import rotations as rot

def _gen_rot(angles, maxdim, device=torch.device('cpu'), dtype=torch.float64, cg_dict=None):

	# save the dictionary of Lorentz-D matrices
	D = {(k, n): rot.LorentzD((k, n), *angles, device=device, dtype=dtype, cg_dict=cg_dict) for k in range(maxdim) for n in range(maxdim)}
	# compute the Lorentz matrix in cartesian coordinates
	cartesian4=torch.tensor([[[1,0,0,0],[0,1/sqrt(2.),0,0],[0,0,0,1],[0,-1/sqrt(2.),0,0]],
                            [[0,0,0,0],[0,0,-1/sqrt(2.),0],[0,0,0,0],[0,0,-1/sqrt(2.),0]]],device=device, dtype=dtype)
	cartesian4H=torch.tensor([[[1,0,0,0],[0,1/sqrt(2.),0,0],[0,0,0,1],[0,-1/sqrt(2.),0,0]],
                            [[0,0,0,0],[0,0,1/sqrt(2.),0],[0,0,0,0],[0,0,1/sqrt(2.),0]]],device=device, dtype=dtype).permute(0,2,1)
	R = torch.stack((D[(1, 1)][0].matmul(cartesian4[0])-D[(1, 1)][1].matmul(cartesian4[1]), D[(1, 1)][0].matmul(cartesian4[1]) + D[(1, 1)][1].matmul(cartesian4[0])))
	R = cartesian4H[0].matmul(R[0]) - cartesian4H[1].matmul(R[1])

	return D, R

def covariance_test(model, data, cg_dict=None):
	logging.info('Beginning covariance test...')
	targets_rotout, outputs_rotin = [], []

	angles = (np.matlib.rand(3) + 1j * np.matlib.rand(3)).tolist()[0]

	device = model.device
	dtype = model.dtype
	data['p4'] = data['p4'].to(device, dtype)
	data['node_mask'] = data['node_mask'].to(device, torch.uint8)
	data['edge_mask'] = data['edge_mask'].to(device, torch.uint8)
	D, R = _gen_rot(angles, model.maxdim, device=device, dtype=dtype, cg_dict=cg_dict)

	# Unrotated input data
	data_rotout = data

	# Rotated input data
	data_rotin = {key: val.clone() if type(val) is torch.Tensor else None for key, val in data.items()}
	data_rotin['p4'] = torch.einsum("...b, ba->...a", data_rotin['p4'], R)

	# Run the model on both inputs
	outputs_rotout, reps_rotout = model(data_rotout, covariance_test=True)
	outputs_rotin, reps_rotin = model(data_rotin, covariance_test=True)

	data['p4']=torch.rand_like(data['p4'])
	data['node_mask'] = torch.ones_like(data['node_mask'])

	logging.info('Boost invariance test. The output is a table of [gamma, relative deviation]:')
	invariance_tests = []
	data_rot = {key: val.clone() if type(val) is torch.Tensor else None for key, val in data.items()}
	alpha_range = np.arange(0,10.0,step=0.5)
	for alpha in alpha_range:
		Di, Ri = _gen_rot((0, 0, alpha*1j), model.maxdim, device=device, dtype=dtype, cg_dict=cg_dict)
		data_rot['p4'] = torch.einsum("...b, ba->...a", data['p4'], Ri)
		outputs_rot, _ = model(data_rot, covariance_test=True)
		invariance_tests.append((outputs_rot))

	boost_result = list(list(x) for x in zip(map(cosh, alpha_range), [((x-invariance_tests[0]).abs().mean()/invariance_tests[0].abs().mean()).abs().item() for x in invariance_tests]))
	logging.info(boost_result)

	logging.info('Rotation invariance test. The output is a table of [angle, rel. deviation]:')
	invariance_tests = []
	data_rot = {key: val.clone() if type(val) is torch.Tensor else None for key, val in data.items()}
	alpha_range = np.arange(0,10.0,step=0.5)
	for alpha in alpha_range:
		Di, Ri = _gen_rot((0, alpha, 0), model.maxdim, device=device, dtype=dtype, cg_dict=cg_dict)
		data_rot['p4'] = torch.einsum("...b, ba->...a", data['p4'], Ri)
		outputs_rot, _ = model(data_rot, covariance_test=True)
		invariance_tests.append((outputs_rot))

	rot_result = list(list(x) for x in zip(alpha_range, [((x-invariance_tests[0]).abs().mean()/invariance_tests[0].abs().mean()).abs().item() for x in invariance_tests]))
	logging.info(rot_result)

	components_mean = [{key: level_in[key].abs().mean().item() for key in level_in.keys()} for level_in in reps_rotin]
	components_median = [{key: level_in[key].abs().median().item() for key in level_in.keys()} for level_in in reps_rotin]

	logging.info("Averages of components of all tensors:\n {}".format(components_mean))
	logging.info("Medians of components of all tensors:\n {}".format(components_median))

	# pin=data_rotin['p4']
	# pout=data_rotout['p4']
	# rin=reps_rotin[1][(1,1)]
	# rout=reps_rotout[1][(1,1)]
	# sin=outputs_rotin
	# sout=outputs_rotout
	# #reps_rotout, reps_rotin = reps_rotout[0], reps_rotin[0]

	# Scalar value measuring the norm of the difference in the outputs
	invariance_test = (outputs_rotout - outputs_rotin).norm()/outputs_rotout.norm()

	# Rotate the intermediate tensors produced by CG levels from the unrotated data
	reps_rotout = [rot.rotate_rep(rep, *angles, cg_dict=cg_dict) for rep in reps_rotout]

	# Measure the difference from the tensors produced from the pre-rotated input
	covariance_test_norm = [{key: (level_in[key] - level_out[key]).norm().item()/level_in[key].norm() for key in level_in.keys()} for (level_in, level_out) in zip(reps_rotin, reps_rotout)]
	covariance_test_mean = [{key: (level_in[key] - level_out[key]).abs().mean().item()/level_in[key].abs().mean() for key in level_in.keys()} for (level_in, level_out) in zip(reps_rotin, reps_rotout)]

	covariance_test_max = torch.cat([torch.tensor([(level_in[key] - level_out[key]).abs().max().item() for key in level_in.keys()]) for (level_in, level_out) in zip(reps_rotin, reps_rotout)])
	covariance_test_max = covariance_test_max.max().item()

	logging.info('Rotation Invariance test (relative): {:0.5g}'.format(invariance_test))
	logging.info('Largest deviation in covariance test : {:0.5g}'.format(covariance_test_max))

	# If the largest deviation in the covariance test is greater than 1e-5,
	# display l1 and l2 norms of the error at each irrep along each level (relative to the norm of the tensor itself).
	if covariance_test_max > 1e-5:
		logging.warning('Largest deviation in covariance test {:0.5g} detected! Detailed summary (all relative):'.format(covariance_test_max))
		for lvl_idx, (lvl_norm, lvl_mean) in enumerate(zip(covariance_test_norm, covariance_test_mean)):
			for key in lvl_norm.keys():
				logging.warning('(lvl, key) = ({}, {}) -> {:0.5g} (norm) {:0.5g} (mean)'.format(lvl_idx, key, lvl_norm[key], lvl_mean[key]))

	return boost_result, rot_result

def permutation_test(model, data):
	logging.info('Beginning permutation test...')

	mask = data['node_mask']

	# Generate a list of indices for each molecule.
	# We will generate a permutation only for the nodes that exist (are not masked.)
	batch_size, node_size = mask.shape
	perm = 1*torch.arange(node_size).expand(batch_size, -1)
	for idx in range(batch_size):
		num_nodes = (mask[idx, :].long()).sum()
		perm[idx, :num_nodes] = torch.randperm(num_nodes)

	apply_perm = lambda mat: torch.stack([mat[idx, p] for (idx, p) in enumerate(perm)])

	assert((mask == apply_perm(mask)).all())

	data_noperm = data
	data_perm = {key: apply_perm(val) if key in ['p4', 'scalars'] else val for key, val in data.items()}

	outputs_perm = model(data_perm)
	outputs_noperm = model(data_noperm)

	invariance_test = (outputs_perm - outputs_noperm).abs().max()/outputs_noperm.abs().max()

	logging.info('Permutation Invariance test error: {}'.format(invariance_test))

	return invariance_test


def batch_test(model, data):
	logging.info('Beginning batch invariance test...')
	data_split = {key: val.unbind(dim=0) if (torch.is_tensor(val) and val.numel() > 1) else val for key, val in data.items()}
	data_split = [{key: val[idx].unsqueeze(0) if type(val) is tuple else val for key, val in data_split.items()} for idx in range(len(data['jet_types']))]

	outputs_split = torch.cat([model(data_sub) for data_sub in data_split])
	outputs_full = model(data)
	invariance_test = (outputs_split - outputs_full).abs().max()/outputs_full.abs().mean()

	logging.info('Batch invariance test error: {}'.format(invariance_test))

	return invariance_test


def lgn_tests(model, dataloader, args, epoch, tests=['covariance','permutation','batch'], cg_dict=None):

	t0 = time.time()

	logging.info("Testing network for symmetries:")
	model.eval()

	lgn_test_results = dict()

	data = next(iter(dataloader))
	if 'covariance' in tests:
		boost_result, rot_result = covariance_test(model, data, cg_dict=cg_dict)
		lgn_test_results['boost_equivariance'] = boost_result
		lgn_test_results['rotation_equivariance'] = rot_result
	if 'permutation' in tests:
		permutation_results = permutation_test(model, data)
		lgn_test_results['permutation_invariance'] = permutation_results
	if 'batch' in tests:
		batch_results = batch_test(model, data)
		lgn_test_results['batch_invariance'] = batch_results

	logging.info('Test complete!')
	t1 = time.time()
	print("Time it took testing equivariance of epoch", epoch, "is:", round((t1-t0)/60,2), "min")

	return lgn_test_results
