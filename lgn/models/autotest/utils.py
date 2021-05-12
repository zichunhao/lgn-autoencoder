def get_output(encoder, decoder, data, covariance_test=False):
	"""
	Get output and all internal features from the autoencoder.
	"""
	latent_features, encoder_nodes_all = encoder(data, covariance_test=covariance_test)
	generated_features, nodes_all = decoder(latent_features, covariance_test=covariance_test, nodes_all=encoder_nodes_all)
	return generated_features, nodes_all

def get_dev(transform_input, transform_output, transform_input_nodes_all, transform_output_nodes_all):
	# Output equivariance
	dev_output = [{weight: ((transform_input[i][weight] - transform_output[i][weight]) / (transform_output[i][weight])).abs().max() for weight in [(0,0), (1,1)]} for i in range(len(transform_output))]
	# Equivariance of all internal features
	dev_internal = [[{weight: ((transform_input_nodes_all[i][j][weight] - transform_output_nodes_all[i][j][weight]) / transform_output_nodes_all[i][j][weight]).abs().max() for weight in [(0,0), (1,1)]} for j in range(len(transform_output_nodes_all[i]))] for i in range(len(transform_output_nodes_all))]
	return dev_output, dev_internal

def get_internal_dev_stats(dev_internal):
    """
    Get mean and max of relative deviation in each layer
    """
    dev_internal_mean = []
    dev_internal_max = []
    for i in range(len(dev_internal)):
        boost_dev_level_scalar = []
        boost_dev_level_p4 = []
        for level in dev_internal[i]:
            boost_dev_level_scalar.append(level[(0,0)].item())
            boost_dev_level_p4.append(level[(1,1)].item())
        dev_internal_mean.append({(0,0): sum(boost_dev_level_scalar) / len(boost_dev_level_scalar), (1,1): sum(boost_dev_level_p4) / len(boost_dev_level_p4)})
        dev_internal_max.append({(0,0): max(boost_dev_level_scalar), (1,1): max(boost_dev_level_p4)})
    return dev_internal_mean, dev_internal_max
