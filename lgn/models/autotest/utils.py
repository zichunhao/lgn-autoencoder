import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import os.path as osp

def get_output(encoder, decoder, data, covariance_test=False):
	"""
	Get output and all internal features from the autoencoder.
	"""
	latent_features, encoder_nodes_all = encoder(data, covariance_test=covariance_test)
	generated_features, nodes_all = decoder(latent_features, covariance_test=covariance_test, nodes_all=encoder_nodes_all)
	return generated_features, nodes_all

def get_dev(transform_input, transform_output, transform_input_nodes_all, transform_output_nodes_all):
	# Output equivariance
	dev_output = [{weight: ((transform_input[i][weight] - transform_output[i][weight]) / (transform_output[i][weight])).abs().max().item() for weight in [(0,0), (1,1)]} for i in range(len(transform_output))]
	# Equivariance of all internal features
	dev_internal = [[{weight: ((transform_input_nodes_all[i][j][weight] - transform_output_nodes_all[i][j][weight]) / transform_output_nodes_all[i][j][weight]).abs().max().item() for weight in [(0,0), (1,1)]} for j in range(len(transform_output_nodes_all[i]))] for i in range(len(transform_output_nodes_all))]
	return dev_output, dev_internal

def get_internal_dev_stats(dev_internal):
	"""
	Get mean and max of relative deviation of all layers
	as well as the deviation in each layer as gamma increases
	"""
	dev_layers = {key: np.array([[dev_internal[i][j][key] for j in range(len(dev_internal[i]))] for i in range(len(dev_internal))]).transpose().tolist() for key in [(0,0),(1,1)]}

	dev_internal_mean = []
	dev_internal_max = []
	for i in range(len(dev_internal)):
		boost_dev_level_scalar = []
		boost_dev_level_p4 = []
		for level in dev_internal[i]:
			boost_dev_level_scalar.append(level[(0,0)])
			boost_dev_level_p4.append(level[(1,1)])
		dev_internal_mean.append({(0,0): sum(boost_dev_level_scalar) / len(boost_dev_level_scalar), (1,1): sum(boost_dev_level_p4) / len(boost_dev_level_p4)})
		dev_internal_max.append({(0,0): max(boost_dev_level_scalar), (1,1): max(boost_dev_level_p4)})

	dev_internal_mean = {key: [dev_internal_mean[i][key] for i in range(len(dev_internal_mean))] for key in [(0,0), (1,1)]}
	dev_internal_max = {key: [dev_internal_max[i][key] for i in range(len(dev_internal_max))] for key in [(0,0), (1,1)]}

	return dev_internal_mean, dev_internal_max, dev_layers

def plot_internal_dev(dev_internal, alphas, transform_type, weight, save_path, show_all=False):
	"""
	Plot internal deviations, mean, and max of all layers.

	Input
	-----
	dev_internal : `list` of `list` of `dict`
		Relative deviations of layers as alpha varies.
		2D-list shape: [len(alphas), num_layers]
		Dict keys: (0,0) and (1,1)
	alphas : `list`
		If transform_type is 'boost', this is the list of Lorentz (boost) factor gammas.
		If transform_type is 'rotation', this is the list of rotation angles.
	transform_type : `str`
		The type of transformation corresponding to the data.
		Choices: ('boost', 'rotation')
	weight : `tuple`
		The weight of the irrep.
		Choices: ((0,0), (1,1))
	show_all : 'bool'
		Whether to show deviations of all layers.
	"""
	make_dir(save_path)
	if weight not in [(0,0), (1,1)]:
		raise ValueError("Weight has to be one of (0,0) and (1,1)")

	dev_internal_mean, dev_internal_max, dev_layers = get_internal_dev_stats(dev_internal)

	irrep_str = '4-vector' if weight == (1,1) else 'scalar'

	if show_all:
		colors = list(plt.cm.tab20(np.arange(len(dev_layers[weight])))) + ["indigo"]
		for i in range(len(dev_layers[weight])):
		    plt.plot(alphas, dev_layers[weight][i], label=f"layer {i+1}", color=colors[i], linewidth=0.9)
		plt.plot(alphas, dev_internal_mean[weight], label="layers mean", color='black', linestyle='dashed', linewidth=1.4)
	if not show_all:
		plt.plot(alphas, dev_internal_mean[weight], label="layers mean")
		plt.plot(alphas, dev_internal_max[weight], label="layers max")

	plt.ylabel('relative deviation')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	if show_all:
		plt.legend(bbox_to_anchor=(1.04,0.85), loc="upper left")
	else:
		plt.legend(loc='best')

	if transform_type.lower() in ['boost', 'boosts']:
		plt.title(f'Boost equivariance test of internal {irrep_str} features', y=1.05)
		plt.xlabel(r'Lorentz factor $\gamma$')
	elif transform_type.lower() in ['rot', 'rots', 'rotation', 'rotatons']:
		plt.title(f'Rotation equivariance test of internal {irrep_str} features', y=1.05)
		plt.xlabel(r'Rotation angle $\theta$ (rad)')

	plt.tight_layout()
	if show_all:
		plt.savefig(osp.join(save_path, f"{transform_type.lower()}_equivariance_test_internal_{irrep_str}_all.pdf"), bbox_inches='tight')
	else:
		plt.savefig(osp.join(save_path, f"{transform_type.lower()}_equivariance_test_internal_{irrep_str}.pdf"), bbox_inches='tight')
	plt.close()

def plot_output_dev(dev_output, alphas, transform_type, weight, save_path):
	make_dir(save_path)
	pkl_path = make_dir(osp.join(save_path, "pkl"))

	if weight not in [(0,0), (1,1)]:
		raise ValueError("Weight has to be one of (0,0) and (1,1)")

	irrep_str = '4-momenta' if weight == (1,1) else 'scalars'
	dev = [dev_output[i][weight] for i in range(len(dev_output))]

	plt.plot(alphas, dev)

	if transform_type.lower() in ['boost', 'boosts']:
		if weight == (1,1):
			title = fr'Boost equivariance test of generated {irrep_str} $p^\mu$'
			torch.save(dev, osp.join(pkl_path, "boost_equivariance_p4.pkl"))
		else:
			title = f'Boost equivariance test of generated {irrep_str}'
			torch.save(dev, osp.join(pkl_path, "boost_equivariance_scalars.pkl"))
		plt.title(title, y=1.05)
		plt.xlabel(r'Lorentz factor $\gamma$')
	elif transform_type.lower() in ['rot', 'rots', 'rotation', 'rotatons']:
		if weight == (1,1):
			title = fr'Rotation equivariance test of generated {irrep_str} $p^\mu$'
			torch.save(dev, osp.join(pkl_path, "rot_equivariance_p4.pkl"))
		else:
			title = f'Rotation equivariance test of generated {irrep_str}'
			torch.save(dev, osp.join(pkl_path, "rot_equivariance_scalars.pkl"))
		plt.title(title, y=1.05)
		plt.xlabel(r'Rotation angle $\theta$ (rad)')

	plt.ylabel('Relative deviation')
	plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.savefig(osp.join(save_path, f"{transform_type.lower()}_equivariance_test_generated_{irrep_str}.pdf"), bbox_inches='tight')
	plt.close()

def plot_all_dev(dev, save_path):
	for weight in [(0,0), (1,1)]:
		plot_output_dev(dev_output=dev['boost_dev_output'], alphas=dev['gammas'], transform_type='boost', weight=weight, save_path=save_path)
		plot_output_dev(dev_output=dev['rot_dev_output'], alphas=dev['thetas'], transform_type='rotation', weight=weight, save_path=save_path)
		for show_option in [True, False]:
			plot_internal_dev(dev_internal=dev['boost_dev_internal'].copy(), alphas=dev['gammas'], transform_type='boost', weight=weight, save_path=save_path, show_all=show_option)
			plot_internal_dev(dev_internal=dev['rot_dev_internal'].copy(), alphas=dev['thetas'], transform_type='rotation', weight=weight, save_path=save_path, show_all=show_option)

def make_dir(path):
    if not osp.isdir(path):
        os.makedirs(path)
    return path
