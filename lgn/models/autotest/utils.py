import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import os.path as osp
import pandas as pd


def get_output(encoder, decoder, data, covariance_test=True):
    """
    Get output and all internal features from the autoencoder.
    """
    latent_features, encoder_nodes_all = encoder(
        data, covariance_test=covariance_test
    )
    generated_features, nodes_all = decoder(
        latent_features, covariance_test=covariance_test, nodes_all=encoder_nodes_all
    )
    return generated_features, nodes_all


def get_node_dev(transform_input, transform_output, eps=1e-16, mode='mean'):
    if mode.lower() == 'max':
        return {
            weight: ((transform_input[weight] - transform_output[weight]) / (transform_output[weight] + eps)).abs().max().item()
            for weight in [(0, 0), (1, 1)]
        }
    else:
        # what is done in [arXiv:2006.04780]
        return {
            weight: (transform_input[weight] - transform_output[weight]).mean().item() / ((transform_output[weight]).mean().item() + eps)
            for weight in [(0, 0), (1, 1)]
        }

def get_dev(transform_input, transform_output, transform_input_nodes_all, transform_output_nodes_all):
    # Output equivariance
    dev_output = [get_node_dev(transform_input[i], transform_output[i])
                  for i in range(len(transform_output))]
    # Equivariance of all internal features
    dev_internal = [[get_node_dev(transform_input_nodes_all[i][j], transform_output_nodes_all[i][j])
                     for j in range(len(transform_output_nodes_all[i]))]
                    for i in range(len(transform_output_nodes_all))]
    return dev_output, dev_internal


def get_avg_output_dev(covariance_results, test_name):
    """
    Get average relative error of output features for each alpha in tests of all epochs.

    Parameter
    ---------
    covariance_results : list of `dict`
        The covariance test results for each epoch. Depending on the test name, the keys of results can be
            - ['gammas', 'boost_dev_output', 'boost_dev_internal']
            - ['thetas', 'rot_dev_output', 'rot_dev_internal']
        We are interested in dicts with the key 'boost_dev_output' or 'rot_dev_output',
        which stores the output relative error by epoch by a 2d-list of `dict` with "shape" (num_epochs, num_gammas).
    test_name : str
        The name of the covariance test. Choices are 'rot' and 'boost'

    Return
    ------
    avg_output_dev_by_alpha : list of `dict`
        The average output relative error of all epochs for all irreps for each alpha (gamma for boost and theta for rotation).
    """

    if test_name.lower() in ['rotation', 'rotations']:
        test_name = 'rot'

    output_devs = [covariance_results[i][f'{test_name.lower()}_dev_output'] for i in range(len(covariance_results))]
    output_devs_by_alpha = np.array(output_devs).transpose().tolist()

    # Adapted from https://stackoverflow.com/a/50782438
    avg_output_dev_by_alpha = [{key: sum(dev[key] for dev in output_devs_by_alpha[i]) / len(output_devs_by_alpha[i])
                                for key in output_devs_by_alpha[i][0].keys()}
                               for i in range(len(output_devs_by_alpha))]

    return avg_output_dev_by_alpha


def get_avg_internal_dev(covariance_results, test_name):
    """
    Get average relative error of internal features for each alpha in tests of all epochs.

    Parameter
    ---------
    covariance_results : list of `dict`
        The covariance test results for each epoch. Depending on the test name, the keys of results can be
            - ['gammas', 'boost_dev_output', 'boost_dev_internal']
            - ['thetas', 'rot_dev_output', 'rot_dev_internal']
        We are interested in dicts with the key 'boost_dev_internal' or 'rot_dev_internal',
        which stores the output relative error by epoch by a 3d-list of `dict` with "shape" (num_epochs, num_gammas, num_model_layers).
    test_name : str
        The name of the covariance test. Choices are 'rot' and 'boost'

    Return
    ------
    avg_output_dev_by_alpha : list of `dict`
        The average internal features relative error of all epochs for all irreps
        for each alpha (gamma for boost and theta for rotation) in each layer.
    """

    if test_name.lower() in ['rotation', 'rotations']:
        test_name = 'rot'

    internal_devs = [covariance_results[i][f'{test_name.lower()}_dev_internal'] for i in range(len(covariance_results))]
    internal_devs_by_alpha = np.array(internal_devs).transpose(1, 2, 0).tolist()

    # Adapted from https://stackoverflow.com/a/50782438
    # Get average
    avg_internal_dev_by_alpha = [
        [
            {
                key: sum(e[key] for e in internal_devs_by_alpha[i][j]) / len(internal_devs_by_alpha[i][j])
                for key in internal_devs_by_alpha[i][j][0].keys()
            }
            for j in range(len(internal_devs_by_alpha[i]))
        ]                        
        for i in range(len(internal_devs_by_alpha))
    ]

    return avg_internal_dev_by_alpha


def get_internal_dev_stats(dev_internal):
    """
    Get mean and max of relative error of all layers
    as well as the deviation in each layer as gamma increases
    """
    dev_layers = {
        key: np.array(
            [
                [dev_internal[i][j][key] for j in range(len(dev_internal[i]))]
                for i in range(len(dev_internal))
            ]
        ).transpose().tolist()
        for key in [(0, 0), (1, 1)]
    }

    dev_internal_mean = []
    dev_internal_max = []
    for i in range(len(dev_internal)):
        boost_dev_level_scalar = []
        boost_dev_level_p4 = []
        for level in dev_internal[i]:
            boost_dev_level_scalar.append(level[(0, 0)])
            boost_dev_level_p4.append(level[(1, 1)])
        dev_internal_mean.append({(0, 0): sum(boost_dev_level_scalar) / len(boost_dev_level_scalar),
                                  (1, 1): sum(boost_dev_level_p4) / len(boost_dev_level_p4)})
        dev_internal_max.append({(0, 0): max(boost_dev_level_scalar),
                                 (1, 1): max(boost_dev_level_p4)})

    dev_internal_mean = {key: [dev_internal_mean[i][key]
                               for i in range(len(dev_internal_mean))] for key in [(0, 0), (1, 1)]}
    dev_internal_max = {key: [dev_internal_max[i][key]
                              for i in range(len(dev_internal_max))] for key in [(0, 0), (1, 1)]}

    return dev_internal_mean, dev_internal_max, dev_layers


def plot_internal_dev(dev_internal, alphas, transform_type, weight, save_path, show_all=False):
    """
    Plot internal deviations, mean, and max of all layers.

    Input
    -----
    dev_internal : list of list of `dict`
            relative errors of layers as alpha varies.
            2D-list shape: [len(alphas), num_layers]
            Dict keys: (0,0) and (1,1)
    alphas : list
            If transform_type is 'boost', this is the list of Lorentz (boost) factor gammas.
            If transform_type is 'rotation', this is the list of rotation angles.
    transform_type : str
            The type of transformation corresponding to the data.
            Choices: ('boost', 'rotation')
    weight : tuple
            The weight of the irrep.
            Choices: ((0,0), (1,1))
    show_all : 'bool'
            Whether to show deviations of all layers.
    """
    make_dir(save_path)
    if weight not in [(0, 0), (1, 1)]:
        raise ValueError("Weight has to be one of (0,0) and (1,1)")

    dev_internal_mean, dev_internal_max, dev_layers = get_internal_dev_stats(dev_internal)

    irrep_str = '4-vector' if weight == (1, 1) else 'scalar'

    if show_all:
        colors = list(plt.cm.tab20(np.arange(len(dev_layers[weight])))) + ["indigo"]
        for i in range(len(dev_layers[weight])):
            plt.plot(alphas, dev_layers[weight][i],
                     label=f"layer {i+1}", color=colors[i], linewidth=0.9)
        plt.plot(alphas, dev_internal_mean[weight], label="layers mean",
                 color='black', linestyle='dashed', linewidth=1.4)
    if not show_all:
        plt.plot(alphas, dev_internal_mean[weight], label="layers mean")
        plt.plot(alphas, dev_internal_max[weight], label="layers max")

    plt.ylabel(r'$\delta_p$')
    try:
        plt.ticklabel_format(axis="y", style="sci", useMathText=True)
    except AttributeError as e:
        # AttributeError: 'LogFormatterSciNotation' object has no attribute 'set_scientific'
        warnings.warn(f"Error in setting ticklabel format: {e}")
        pass

    if show_all:
        plt.legend(bbox_to_anchor=(1.04, 0.85), loc="upper left")
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
        plt.savefig(osp.join(save_path, f"{transform_type.lower()}_equivariance_test_internal_{irrep_str}_all.pdf"),
                    bbox_inches='tight', transparent=True)
    else:
        plt.savefig(osp.join(save_path, f"{transform_type.lower()}_equivariance_test_internal_{irrep_str}.pdf"),
                    bbox_inches='tight', transparent=True)
    plt.close()


def plot_output_dev(dev_output, alphas, transform_type, weight, save_path):
    make_dir(save_path)
    pt_path = make_dir(osp.join(save_path, "pt_files"))

    if weight not in [(0, 0), (1, 1)]:
        raise ValueError("Weight has to be one of (0,0) and (1,1)")

    irrep_str = '4-momenta' if weight == (1, 1) else 'scalars'
    dev = [dev_output[i][weight] for i in range(len(dev_output))]

    plt.plot(alphas, dev)

    if transform_type.lower() in ['boost', 'boosts']:
        if weight == (1, 1):
            title = fr'Boost equivariance test of reconstructed {irrep_str} $p^\mu$'
            torch.save(dev, osp.join(pt_path, "boost_equivariance_p4.pt"))
        else:
            title = f'Boost equivariance test of reconstructed {irrep_str}'
            torch.save(dev, osp.join(pt_path, "boost_equivariance_scalars.pt"))
        plt.title(title, y=1.05)
        plt.xlabel(r'Lorentz factor $\gamma$')
    elif transform_type.lower() in ['rot', 'rots', 'rotation', 'rotations']:
        if weight == (1, 1):
            title = fr'Rotation equivariance test of reconstructed {irrep_str} $p^\mu$'
            torch.save(dev, osp.join(pt_path, "rot_equivariance_p4.pt"))
        else:
            title = f'Rotation equivariance test of reconstructed {irrep_str}'
            torch.save(dev, osp.join(pt_path, "rot_equivariance_scalars.pt"))
        plt.title(title, y=1.05)
        plt.xlabel(r'Rotation angle $\theta$ (rad)')

    plt.ylabel(r'$\delta_p$')
    if weight == (1, 1):
        plt.yscale('log')
        try:
            plt.ticklabel_format(axis="y", style="sci", useMathText=True)
        except AttributeError as e:
             # AttributeError: 'LogFormatterSciNotation' object has no attribute 'set_scientific'
            warnings.warn(f"Error in setting ticklabel format: {e}")
            pass

    plt.savefig(osp.join(save_path, f"{transform_type.lower()}_equivariance_test_reconstructed_{irrep_str}.pdf"),
                bbox_inches='tight', transparent=True)
    plt.close()


def plot_all_dev(dev, save_path):
    make_dir(save_path)
    pt_path = make_dir(osp.join(save_path, "pt_files"))
    torch.save(dev['perm_invariance_dev_output'], osp.join(pt_path, "perm_invariance_dev_output.pt"))
    torch.save(dev['perm_equivariance_dev_output'], osp.join(pt_path, "perm_equivariance_dev_output.pt"))

    for weight in [(0, 0), (1, 1)]:
        plot_output_dev(dev_output=dev['boost_dev_output'], alphas=dev['gammas'],
                        transform_type='boost', weight=weight, save_path=save_path)
        plot_output_dev(dev_output=dev['rot_dev_output'], alphas=dev['thetas'],
                        transform_type='rotation', weight=weight, save_path=save_path)
        for show_option in [True, False]:
            plot_internal_dev(dev_internal=dev['boost_dev_internal'].copy(),
                              alphas=dev['gammas'], transform_type='boost', weight=weight,
                              save_path=save_path, show_all=show_option)
            plot_internal_dev(dev_internal=dev['rot_dev_internal'].copy(),
                              alphas=dev['thetas'], transform_type='rotation', weight=weight,
                              save_path=save_path, show_all=show_option)


def make_dir(path):
    if not osp.isdir(path):
        os.makedirs(path)
    return path


def display_err(alphas, errs, alpha_name, caption):
    err_dict = dict()
    err_dict[alpha_name] = alphas
    for k in errs[0].keys():
        err_dict[str(k)] = [errs[i][k] for i in range(len(errs))]
    df = pd.DataFrame(err_dict)
    df.style.set_table_attributes("style='display:inline'").set_caption(caption)
    print(df)
    return df
