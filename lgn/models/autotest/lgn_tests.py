import torch
import numpy as np
import time
import numpy.matlib
from math import sqrt, cosh
import logging

from lgn.g_lib import rotations as rot
from lgn.models.autotest.utils import get_output, get_dev, get_avg_output_dev, get_avg_internal_dev, get_node_dev


def _gen_rot(angles, maxdim, device=torch.device('cpu'), dtype=torch.float64, cg_dict=None):

    # save the dictionary of Lorentz-D matrices
    D = {(k, n): rot.LorentzD((k, n), *angles, device=device, dtype=dtype, cg_dict=cg_dict)
         for k in range(maxdim) for n in range(maxdim)}
    # compute the Lorentz matrix in cartesian coordinates
    cartesian4 = torch.tensor([[[1, 0, 0, 0], [0, 1/sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1/sqrt(2.), 0, 0]],
                               [[0, 0, 0, 0], [0, 0, -1/sqrt(2.), 0], [0, 0, 0, 0], [0, 0, -1/sqrt(2.), 0]]], device=device, dtype=dtype)
    cartesian4H = torch.tensor([[[1, 0, 0, 0], [0, 1/sqrt(2.), 0, 0], [0, 0, 0, 1], [0, -1/sqrt(2.), 0, 0]],
                                [[0, 0, 0, 0], [0, 0, 1/sqrt(2.), 0], [0, 0, 0, 0], [0, 0, 1/sqrt(2.), 0]]], device=device, dtype=dtype).permute(0, 2, 1)
    R = torch.stack((D[(1, 1)][0].matmul(cartesian4[0])-D[(1, 1)][1].matmul(cartesian4[1]),
                     D[(1, 1)][0].matmul(cartesian4[1]) + D[(1, 1)][1].matmul(cartesian4[0])))
    R = cartesian4H[0].matmul(R[0]) - cartesian4H[1].matmul(R[1])

    return D, R


def covariance_test(encoder, decoder, data, test_type, alpha_max=None, cg_dict=None):

    if cg_dict is None:
        cg_dict = encoder.cg_dict

    device = encoder.device
    dtype = encoder.dtype
    data['p4'] = data['p4'].to(device, dtype)
    # data['p4'] = torch.rand_like(data['p4'])

    covariance_test_result = dict()

    if test_type.lower() in ['boost', 'boosts']:
        if alpha_max is None:
            alpha_max = 10.
        alpha_range = np.arange(0, alpha_max+.01, step=alpha_max/25.)
        gammas, boost_dev_output, boost_dev_internal = boost_equivariance(encoder, decoder, data, alpha_range, device, dtype, cg_dict)
        covariance_test_result['gammas'] = gammas
        covariance_test_result['boost_dev_output'] = boost_dev_output
        covariance_test_result['boost_dev_internal'] = boost_dev_internal

    elif test_type.lower() in ['rot', 'rotation', 'rotations']:
        if alpha_max is None:
            alpha_max = 2 * np.pi
        theta_range = np.arange(0, alpha_max+.01, step=alpha_max/25.)
        theta_range, rot_dev_output, rot_dev_internal = rot_equivariance(encoder, decoder, data, theta_range, device, dtype, cg_dict)
        covariance_test_result['thetas'] = theta_range
        covariance_test_result['rot_dev_output'] = rot_dev_output
        covariance_test_result['rot_dev_internal'] = rot_dev_internal

    else:
        raise ValueError(f"test_type must be one of 'boost' or 'rotation': {test_type}")

    return covariance_test_result


def permutation_invariance_test(encoder, decoder, data, *ignore):
    mask = data['labels']
    batch_size, node_size = mask.shape
    perm = 1*torch.arange(node_size).expand(batch_size, -1)

    for idx in range(batch_size):
        num_nodes = (mask[idx, :].long()).sum()
        perm[idx, :num_nodes] = torch.randperm(num_nodes)

    def apply_perm(mat):
        return torch.stack([mat[idx, p] for (idx, p) in enumerate(perm)])

    assert((mask == apply_perm(mask)).all())

    data_noperm = data.copy()
    data_perm = {key: apply_perm(val) if key in ['p4', 'scalars'] else val for key, val in data.items()}

    outputs_perm, internal_perm = get_output(encoder, decoder, data_perm, covariance_test=True)
    outputs_noperm, internal = get_output(encoder, decoder, data_noperm, covariance_test=True)

    output_dev = get_node_dev(outputs_perm, outputs_noperm)

    return output_dev


def boost_equivariance(encoder, decoder, data, alpha_range, device, dtype, cg_dict):
    gammas = list(cosh(x) for x in alpha_range)
    boost_input = []
    boost_output = []
    boost_input_nodes_all = []
    boost_output_nodes_all = []
    for alpha in alpha_range:
        # boost input
        Di, Ri = _gen_rot((0, 0, alpha*1j), encoder.maxdim,
                          device=device, dtype=dtype, cg_dict=cg_dict)
        data_boost = data.copy()
        data_boost['p4'] = torch.einsum("...b, ba->...a", data['p4'], Ri)  # Boost input
        res_boost_input, internal_boost_input = get_output(
            encoder, decoder, data_boost, covariance_test=True)
        boost_input.append((res_boost_input))
        boost_input_nodes_all.append((internal_boost_input))

        # boost output
        res, internal = get_output(encoder, decoder, data, covariance_test=True)
        boost_res = rot.rotate_rep(res, 0, 0, alpha*1j, cg_dict=cg_dict)
        boost_internal = [rot.rotate_rep(internal[i], 0, 0, alpha*1j, cg_dict=cg_dict)
                          for i in range(len(internal))]
        boost_output.append((boost_res))
        boost_output_nodes_all.append((boost_internal))

        dev_output, dev_internal = get_dev(
            boost_input, boost_output, boost_input_nodes_all, boost_output_nodes_all)

    return gammas, dev_output, dev_internal


def rot_equivariance(encoder, decoder, data, theta_range, device, dtype, cg_dict):
    rot_input = []
    rot_output = []
    rot_input_nodes_all = []
    rot_output_nodes_all = []
    for theta in theta_range:
        # rotate input -> output
        Di, Ri = _gen_rot((0, theta, 0), encoder.maxdim,
                          device=device, dtype=dtype, cg_dict=cg_dict)
        data_boost = data.copy()
        data_boost['p4'] = torch.einsum("...b, ba->...a", data['p4'], Ri)  # Rotate input
        res_rot_input, internal_rot_input = get_output(
            encoder, decoder, data_boost, covariance_test=True)
        rot_input.append((res_rot_input))
        rot_input_nodes_all.append((internal_rot_input))

        # Input -> rotate output
        res, internal = get_output(encoder, decoder, data, covariance_test=True)
        rot_res = rot.rotate_rep(res, 0, theta, 0, cg_dict=cg_dict)
        rot_internal = [rot.rotate_rep(internal[i], 0, theta, 0, cg_dict=cg_dict)
                        for i in range(len(internal))]
        rot_output.append((rot_res))
        rot_output_nodes_all.append((rot_internal))

        dev_output, dev_internal = get_dev(
            rot_input, rot_output, rot_input_nodes_all, rot_output_nodes_all)

    return theta_range, dev_output, dev_internal


def lgn_tests(encoder, decoder, dataloader, args, epoch, alpha_max=None, theta_max=None, cg_dict=None):

    t0 = time.time()

    logging.info("Covariance test begins...")
    encoder.eval()
    decoder.eval()

    boost_test_all_epochs = []
    rot_test_all_epochs = []
    perm_test_all_epochs = []

    for data in dataloader:
        boost_results = covariance_test(encoder, decoder, data, test_type='boost',
                                        cg_dict=cg_dict, alpha_max=alpha_max)
        boost_test_all_epochs.append(boost_results)

        rot_results = covariance_test(encoder, decoder, data, test_type='rotation',
                                      cg_dict=cg_dict, alpha_max=theta_max)
        rot_test_all_epochs.append(rot_results)

        perm_result = permutation_invariance_test(encoder, decoder, data)
        perm_test_all_epochs.append(perm_result)

    dt = time.time() - t0
    logging.info(f"Covariance test completed! Time it took testing equivariance of epoch {epoch+1} is {round(dt/60, 2)} min")

    lgn_test_results = dict()

    lgn_test_results['gammas'] = boost_test_all_epochs[0]['gammas']
    lgn_test_results['boost_dev_output'] = get_avg_output_dev(boost_test_all_epochs, 'boost')
    lgn_test_results['boost_dev_internal'] = get_avg_internal_dev(boost_test_all_epochs, 'boost')

    lgn_test_results['thetas'] = rot_test_all_epochs[0]['thetas']
    lgn_test_results['rot_dev_output'] = get_avg_output_dev(rot_test_all_epochs, 'rotation')
    lgn_test_results['rot_dev_internal'] = get_avg_internal_dev(rot_test_all_epochs, 'rotation')

    perm_test_avg = {key: sum(dev[key] for dev in perm_test_all_epochs) / len(perm_test_all_epochs)
                     for key in perm_test_all_epochs[0].keys()}
    lgn_test_results['perm_dev_output'] = perm_test_avg

    logging.info(f"Permutation invaraince test result: {lgn_test_results['perm_dev_output']}")

    return lgn_test_results
