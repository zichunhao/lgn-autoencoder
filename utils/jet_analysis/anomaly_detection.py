from sklearn import metrics
from typing import Dict, List, Tuple, Union
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from .utils import arcsinh
from pathlib import Path
from scipy import optimize

import torch
import energyflow
import numpy as np

EPS_DEFAULT = 1e-16
# keys for scores
CHAMFER_PARTICLE_CARTESIAN = "particle, Cartesian, Chamfer distance"
CHAMFER_PARTICLE_POLAR = "particle, polar, Chamfer distance"
CHAMFER_PARTICLE_NORMALIZED_CARTESIAN = "particle, normalized Cartesian, Chamfer distance"
CHAMFER_PARTICLE_NORMALIZED_POLAR = "particle, normalized polar, Chamfer distance"
CHAMFER_PARTICLE_RELATIVE_POLAR = "particle, relative polar, Chamfer distance"

HUNGARIAN_PARTICLE_CARTESIAN = "particle, Cartesian, Hungarian distance"
HUNGARIAN_PARTICLE_POLAR = "particle, polar, Hungarian distance"
HUNGARIAN_PARTICLE_NORMALIZED_CARTESIAN = "particle, normalized Cartesian, Hungarian distance"
HUNGARIAN_PARTICLE_NORMALIZED_POLAR = "particle, normalized polar, Hungarian distance"
HUNGARIAN_PARTICLE_RELATIVE_POLAR = "particle, relative polar, Hungarian distance"

MSE_PARTICLE_CARTESIAN = "particle, Cartesian, MSE"
MSE_PARTICLE_POLAR = "particle, polar, MSE"
MSE_PARTICLE_NORMALIZED_CARTESIAN = "particle, normalized Cartesian, MSE"
MSE_PARTICLE_NORMALIZED_POLAR = "particle, normalized polar, MSE"
MSE_PARTICLE_RELATIVE_POLAR = "particle, relative polar, MSE"

JET_CARTESIAN = "jet, Cartesian"
JET_POLAR = "jet, polar"
MSE_PARTICLE_LORENTZ = "particle, Lorentz norms, MSE"
CHAMFER_PARTICLE_LORENTZ = "particle, Lorentz norms, Chamfer distance"
HUNGARIAN_PARTICLE_LORENTZ = "particle, Lorentz norms, Hungarian distance"
JET_LORENTZ = "jet, Lorentz norms"
EMD = 'emd'
EMD_RELATIVE = 'emd (relative coordinates)'


def get_ROC_AUC(
    scores_dict: Dict[str, np.ndarray],
    true_labels: np.ndarray,
    save_path: Union[str, Path] = None,
    plot_rocs: bool = True,
    rocs_hlines: List[float] = [1e-1, 1e-2]
) -> Tuple[Dict[str, Tuple[np.ndarray]], Dict[str, Tuple[np.ndarray]]]:
    """Get AUC and ROC curves given scores and true labels.

    :param scores_dict: Dictionary of scores. Keys: metric names, values: scores.
    :type scores_dict: Dict[str, np.ndarray]
    :param true_labels: True labels.
    :type true_labels: np.ndarray
    :param save_path: Path to save the ROC curves and AUCs, defaults to None. 
    If None, the ROC curves and AUCs are not saved.
    :type save_path: str, optional
    :param rocs_hlines: Horizontal lines (and intercept) to plot on the ROC curves, 
    defaults to [1e-1, 1e-2].
    :type rocs_hlines: List[float], optional
    :return: (`roc_curves`, `aucs`), 
    where `roc_curves` is a dictionary {kind: roc_curve}, 
    and `aucs` is a dictionary {kind: auc}.
    :rtype: Tuple[Dict[str, Tuple[np.ndarray]], Dict[str, Tuple[np.ndarray]]]
    """
    roc_curves = dict()
    aucs = dict()
    for kind, scores in scores_dict.items():
        roc_curve = metrics.roc_curve(true_labels, scores)
        roc_curves[kind] = roc_curve
        auc = metrics.auc(roc_curve[0], roc_curve[1])
        if auc < 0.5:
            # opposite labels
            roc_curve = metrics.roc_curve(-true_labels, scores)
            roc_curves[kind] = roc_curve
            auc = metrics.auc(roc_curve[0], roc_curve[1])
        aucs[kind] = auc

    if save_path is not None:
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(scores_dict, save_path / 'scores.pt')
        torch.save(true_labels, save_path / 'true_labels.pt')
        torch.save(roc_curves, save_path / 'roc_curves.pt')
        torch.save(aucs, save_path / 'aucs.pt')

    if plot_rocs:
        auc_sorted = list(sorted(aucs.items(), key=lambda x: x[1], reverse=True))

        if save_path is not None:
            plot_roc_curves(
                aucs=auc_sorted,
                roc_curves=roc_curves,
                rocs_hlines=rocs_hlines,
                path=save_path / 'roc_curves.pdf'
            )
            plot_roc_curves(
                aucs=auc_sorted[:3],
                roc_curves=roc_curves,
                rocs_hlines=rocs_hlines,
                path=save_path / 'roc_curves_top3.pdf'
            )
            plot_roc_curves(
                aucs=auc_sorted[:1],
                roc_curves=roc_curves,
                rocs_hlines=rocs_hlines,
                path=save_path / 'roc_curves_top1.pdf'
            )
        else:
            plot_roc_curves(
                auc_sorted, 
                roc_curves=roc_curves,
                rocs_hlines=rocs_hlines,
                path=None
            )

    return roc_curves, aucs

def plot_roc_curves(
    aucs: Tuple[str, float], 
    roc_curves: Dict[str, np.ndarray],
    rocs_hlines: List[float] = [1e-1, 1e-2],
    path: Union[str, Path] = None
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_xlabel('True Positive Rate')
    ax.set_ylabel('False Positive Rate')
    ax.set_yscale('log')

    for kind, auc in aucs:

        fpr, tpr, thresholds = roc_curves[kind]
        ax.plot(tpr, fpr, label=f'{kind} (AUC: {auc:.5f})')
        for y_value in rocs_hlines:
            ax.plot(
                np.linspace(0, 1, 100), [y_value] * 100,
                '--', c='gray', linewidth=1
            )
            ax.vlines(
                x=tpr[np.searchsorted(fpr, y_value)],
                ymin=0, ymax=y_value,
                linestyles="--", colors="gray", linewidth=1
            )
        fpr, tpr, thresholds = roc_curves[kind]

    plt.legend()
    if path is not None:
        plt.savefig(path)

    return fig


def anomaly_scores_sig_bkg(
    sig_recons: torch.Tensor,
    sig_target: torch.Tensor,
    sig_recons_normalized: torch.Tensor,
    sig_target_normalized: torch.Tensor,
    bkg_recons: torch.Tensor,
    bkg_target: torch.Tensor,
    bkg_recons_normalized: torch.Tensor,
    bkg_target_normalized: torch.Tensor,
    include_emd: bool = True,
    batch_size: int = -1
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute anomaly scores for signal and background
    and return the anomaly scores along with the true labels.

    :param sig_recons: Reconstructed signal jets.
    :type sig_recons: torch.Tensor
    :param sig_target: Target signal jets.
    :type sig_target: torch.Tensor
    :param sig_recons_normalized: Reconstructed normalized signal jets.
    :type sig_recons_normalized: torch.Tensor
    :param sig_target_normalized: Reconstructed normalized signal jets.
    :type sig_target_normalized: torch.Tensor
    :param bkg_recons: Reconstructed background jets.
    :type bkg_recons: torch.Tensor
    :param bkg_target: Target background jets.
    :type bkg_target: torch.Tensor
    :param bkg_recons_normalized: Reconstructed normalized background jets.
    :type bkg_recons_normalized: torch.Tensor
    :param bkg_target_normalized: Target normalized background jets.
    :type bkg_target_normalized: torch.Tensor
    :param include_emd: Whether to include EMD loss as score, defaults to True
    :type include_emd: bool, optional
    :param batch_size: Batch size, defaults to -1.
    If it is a non-positive number or None, then the data will no be batched. 
    :type batch_size: int, optional
    :return: (scores, true_labels, sig_scores, bkg_scores), 
    where scores is a dictionary 
    with the scores (value) for each type (key).
    :rtype: Tuple[np.ndarray, Dict[str, np.ndarray]]
    """
    sig_scores = anomaly_scores(
        sig_recons,
        sig_target,
        sig_recons_normalized,
        sig_target_normalized,
        include_emd=include_emd,
        batch_size=batch_size
    )
    bkg_scores = anomaly_scores(
        bkg_recons,
        bkg_target,
        bkg_recons_normalized,
        bkg_target_normalized,
        include_emd=include_emd,
        batch_size=batch_size
    )
    scores = {
        k: np.concatenate([sig_scores[k], bkg_scores[k]])
        for k in sig_scores.keys()
    }
    true_labels = np.concatenate([
        np.ones_like(sig_scores[CHAMFER_PARTICLE_CARTESIAN]),
        -np.ones_like(bkg_scores[CHAMFER_PARTICLE_CARTESIAN])
    ])
    return scores, true_labels, sig_scores, bkg_scores


def anomaly_scores(
    recons: torch.Tensor,
    target: torch.Tensor,
    recons_normalized: torch.Tensor,
    target_normalized: torch.Tensor,
    include_emd: bool = True,
    batch_size: int = -1
) -> Dict[str, np.ndarray]:
    """Get anomaly scores for a batch of jets.

    :param recons: Reconstructed jets.
    :type recons: torch.Tensor
    :param target: Target jets.
    :type target: torch.Tensor
    :param recons_normalized: Normalized reconstructed jets.
    :type recons_normalized: torch.Tensor
    :param target_normalized: Normalized target jets.
    :type target_normalized: torch.Tensor
    :param include_emd: Whether to include EMD loss as a score, defaults to True
    :type include_emd: bool, optional
    :param batch_size: Batch size, defaults to -1.
    If it is a non-positive number or None, then the data will no be batched. 
    :type batch_size: int, optional
    :return: A dictionary with the scores (value) for each type (key).
    :rtype: Dict[str, np.ndarray]
    """

    # prepare inputs
    recons_polar = get_p4_polar(recons)
    target_polar = get_p4_polar(target)

    recons_normalized_polar = get_p4_polar(recons_normalized)
    target_normalized_polar = get_p4_polar(target_normalized)

    recons_jet = get_jet_p4(recons)
    target_jet = get_jet_p4(target)
    recons_jet_polar = get_p4_polar(recons_jet)
    target_jet_polar = get_p4_polar(target_jet)

    target_polar_rel = get_polar_rel(target_polar, target_jet_polar)
    recons_polar_rel = get_polar_rel(recons_polar, recons_jet_polar)

    recons_jet = recons_jet.view(-1, 4)
    target_jet = target_jet.view(-1, 4)
    recons_jet_polar = recons_jet_polar.view(-1, 4)
    target_jet_polar = target_jet_polar.view(-1, 4)

    scores = {
        # average over jets
        # Chamfer
        CHAMFER_PARTICLE_CARTESIAN: chamfer(recons, target, batch_size=batch_size).mean(-1).numpy(),
        CHAMFER_PARTICLE_POLAR: chamfer(recons_polar, target_polar, batch_size=batch_size).mean(-1).numpy(),
        CHAMFER_PARTICLE_NORMALIZED_CARTESIAN: chamfer(recons_normalized, target_normalized, batch_size=batch_size).mean(-1).numpy(),
        CHAMFER_PARTICLE_NORMALIZED_POLAR: chamfer(recons_normalized_polar, target_normalized_polar, batch_size=batch_size).mean(-1).numpy(),
        CHAMFER_PARTICLE_RELATIVE_POLAR: chamfer(recons_polar_rel, target_polar_rel, batch_size=batch_size).mean(-1).numpy(),
        # Hungarian (linear assignment)
        HUNGARIAN_PARTICLE_CARTESIAN: hungarian(recons, target, batch_size=batch_size).mean(-1).numpy(),
        HUNGARIAN_PARTICLE_POLAR: hungarian(recons_polar, target_polar, batch_size=batch_size).mean(-1).numpy(),
        HUNGARIAN_PARTICLE_NORMALIZED_CARTESIAN: hungarian(recons_normalized, target_normalized, batch_size=batch_size).mean(-1).numpy(),
        HUNGARIAN_PARTICLE_NORMALIZED_POLAR: hungarian(recons_normalized_polar, target_normalized_polar, batch_size=batch_size).mean(-1).numpy(),
        HUNGARIAN_PARTICLE_RELATIVE_POLAR: hungarian(recons_polar_rel, target_polar_rel, batch_size=batch_size).mean(-1).numpy(),
        # MSE
        MSE_PARTICLE_CARTESIAN: mse(recons, target).mean(-1).numpy(),
        MSE_PARTICLE_POLAR: mse(recons_polar, target_polar).mean(-1).numpy(),
        MSE_PARTICLE_NORMALIZED_CARTESIAN: mse(recons_normalized, target_normalized).mean(-1).numpy(),
        MSE_PARTICLE_NORMALIZED_POLAR: mse(recons_normalized_polar, target_normalized_polar).mean(-1).numpy(),
        MSE_PARTICLE_RELATIVE_POLAR: mse(recons_polar_rel, target_polar_rel).mean(-1).numpy(),
        # metrics based on jets
        JET_CARTESIAN: mse(recons_jet, target_jet).numpy(),
        JET_POLAR: mse(recons_jet, target_jet).numpy(),
        # Lorentz invariant scores
        CHAMFER_PARTICLE_LORENTZ: chamfer_lorentz(recons, target, batch_size=batch_size).mean(-1).numpy(),
        HUNGARIAN_PARTICLE_LORENTZ: hungarian_lorentz(recons, target, batch_size=batch_size).mean(-1).numpy(),
        MSE_PARTICLE_LORENTZ: mse_lorentz(recons, target).mean(-1).numpy(),
        JET_LORENTZ: mse_lorentz(recons_jet, target_jet).numpy()
    }

    if include_emd:
        scores[EMD] = emd_loss(recons_polar, target_polar)
        scores[EMD_RELATIVE] = emd_loss(target_polar_rel, recons_polar_rel)

    return scores


# Helper functions
def norm_sq_Lorentz(x: torch.Tensor) -> torch.Tensor:
    E, px, py, pz = x.unbind(-1)
    return E**2 - px**2 - py**2 - pz**2


def mse_lorentz(
    p: torch.Tensor,
    q: torch.Tensor
) -> torch.Tensor:
    """MSE Loss using Lorentzian metric.

    :param p: Output tensor.
    :type p: torch.Tensor
    :param q: Target tensor.
    :type q: torch.Tensor
    :return: MSE Loss between p and q using Lorentzian metric.
    :rtype: torch.Tensor
    """
    return norm_sq_Lorentz(p - q)


def emd_loss(
    recons_polar: torch.Tensor,
    target_polar: torch.Tensor
) -> np.ndarray:
    """Get EMD loss between reconstructed and target jets 
    in polar coordinates (E, pt, eta, phi) or (pt, eta, phi).

    :param recons_polar: Reconstructed jets in polar coordinates.
    :type recons_polar: torch.Tensor
    :param target_polar: Target jets in polar coordinates.
    :type target_polar: torch.Tensor
    :raises ValueError: if the shape of the reconstructed and target jets.
    :return: The EMD loss between the reconstructed and target jets.
    :rtype: np.ndarray
    """
    if recons_polar.shape != target_polar.shape:
        raise ValueError(
            f'recons_polar and target must have the same shape. '
            f"Got: {recons_polar.shape=} and {target_polar.shape=}."
        )

    def emd_loss_jet(p_polar: np.ndarray, q_polar: np.ndarray) -> np.ndarray:
        if p_polar.shape[-1] == 4:
            p_polar = p_polar[..., 1:]
        if q_polar.shape[-1] == 4:
            q_polar = q_polar[..., 1:]
        # (pT, eta, phi): https://energyflow.network/docs/emd/#emd
        return energyflow.emd.emd(p_polar.numpy(), q_polar.numpy())

    losses = []
    for i in range(target_polar.shape[0]):
        p, q = recons_polar[i], target_polar[i]
        losses.append(emd_loss_jet(p, q))
    return np.array(losses)


def mse(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    return ((p - q)**2).sum(dim=dim)


def chamfer(
    p: torch.Tensor,
    q: torch.Tensor,
    batch_size: int = -1
) -> torch.Tensor:
    """Compute the chamfer distance between two batched data.

    :param p: First tensor.
    :type p: torch.Tensor
    :param q: Second tensor.
    :type q: torch.Tensor
    :param batch_size: Batch size, defaults to -1.
    If it is a non-positive number or None, then the data will no be batched. 
    :type batch_size: int, optional
    :return: _description_
    :rtype: torch.Tensor
    """
    if (batch_size is not None) and (batch_size > 0):
        # batched version
        dataset = DistanceDataset(p, q)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        chamfer_dists = []
        for p, q in dataloader:
            # call non-batched version
            chamfer_dists.append(chamfer(p, q, batch_size=-1))
        return torch.cat(chamfer_dists, dim=0)
    else:
        # non-batched version
        diffs = torch.unsqueeze(p, -2) - torch.unsqueeze(q, -3)
        dist = torch.norm(diffs, dim=-1)
        min_dist_pq = torch.min(dist, dim=-1).values
        min_dist_qp = torch.min(dist, dim=-2).values
        return min_dist_pq + min_dist_qp


def chamfer_lorentz(
    p: torch.Tensor,
    q: torch.Tensor,
    batch_size: int = -1
) -> torch.Tensor:
    if (batch_size is not None) and (batch_size > 0):
        # batched version
        dataset = DistanceDataset(p, q)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        chamfer_dists = []
        for p, q in dataloader:
            # call non-batched version
            chamfer_dists.append(chamfer_lorentz(p, q, batch_size=-1))
        return torch.cat(chamfer_dists, dim=0)
    else:
        diffs = torch.unsqueeze(p, -2) - torch.unsqueeze(q, -3)
        dist = norm_sq_Lorentz(diffs)
        min_dist_pq = torch.min(dist, dim=-1).values
        min_dist_qp = torch.min(dist, dim=-2).values
        return min_dist_pq + min_dist_qp
    
def hungarian(
    p: torch.Tensor, 
    q: torch.Tensor, 
    batch_size: int = -1
) -> torch.Tensor:
    """Get the Hungarian distance between two batched data.

    :param p: Reconstructed jets.
    :type p: torch.Tensor
    :param q: Target jets.
    :type q: torch.Tensor
    :param batch_size: Batch size when computing the distances, defaults to -1.
    When -1 or None, the data will not be batched.
    :type batch_size: int, optional
    :return: Hungarian distance between p and q.
    :rtype: torch.Tensor
    """
    if (batch_size is not None) and (batch_size > 0):
        # batched version
        dataset = DistanceDataset(p, q)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        hungarian_distances = []
        for p, q in dataloader:
            # call non-batched version
            hungarian_distances.append(hungarian(p, q, batch_size=-1))
        return torch.cat(hungarian_distances, dim=0)
    else:
        # non-batched version (base case)
        cost = torch.cdist(p, q).cpu().detach().numpy()
        matching = [
            optimize.linear_sum_assignment(cost[i])[1] 
            for i in range(len(cost))
        ]

        p_shuffle = torch.zeros(p.shape).to(p.device).to(p.dtype)
        for i in range(len(matching)):
            p_shuffle[i] = p[i, matching[i]]
        return mse(p_shuffle, q)
    
def hungarian_lorentz(
    p: torch.Tensor,
    q: torch.Tensor,
    batch_size: int = -1
) -> torch.Tensor:
    """Get the Hungarian distance between two batched data 
    in terms of the Minkowskian metric diag(+, -, -, -).

    :param p: Reconstructed jets.
    :type p: torch.Tensor
    :param q: Target jets.
    :type q: torch.Tensor
    :param batch_size: Batch size when computing the distances, defaults to -1.
    When -1 or None, the data will not be batched.
    :type batch_size: int, optional
    :return: Hungarian distance between p and q
    in terms of the Minkowskian metric diag(+, -, -, -).
    :rtype: torch.Tensor
    """
    if (batch_size is not None) and (batch_size > 0):
        # batched version
        dataset = DistanceDataset(p, q)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        hungarian_distances = []
        for p, q in dataloader:
            # call non-batched version
            hungarian_distances.append(hungarian_lorentz(p, q, batch_size=-1))
        return torch.cat(hungarian_distances, dim=0)
    else:
        # non-batched version (base case)
        diffs = torch.unsqueeze(p, -2) - torch.unsqueeze(q, -3)
        cost = norm_sq_Lorentz(diffs).cpu().detach().numpy()
        matching = [
            optimize.linear_sum_assignment(cost[i])[1] 
            for i in range(len(cost))
        ]

        p_shuffle = torch.zeros(p.shape).to(p.device).to(p.dtype)
        for i in range(len(matching)):
            p_shuffle[i] = p[i, matching[i]]
        return mse(p_shuffle, q)


def normalize_particle_features(
    p: torch.Tensor,
    eps: float = EPS_DEFAULT
) -> torch.Tensor:
    """Normalize by dividing the largest ..math::`p_i` 
    (in terms of the absolute value) in the jet 
    for each ..math::`i` (where ..math::`i \in \{ 0, 1, 2, 3 \}`)

    :param p: Particle features.
    :type p: torch.Tensor
    :param eps: Epsilon value to use for division to avoid ZeroDivisionError,
    defaults to 1e-16.
    :type eps: float, optional
    :return: Normalized particle features.
    :rtype: torch.Tensor
    """
    norm_factor = torch.abs(p).amax(dim=-2, keepdim=True)
    return p / (norm_factor + eps)


def get_p4_polar(
    p: torch.Tensor,
    eps: float = EPS_DEFAULT
) -> torch.Tensor:
    """(E, px, py, pz) -> (E, pT, eta, phi)"""
    E, px, py, pz = p.unbind(-1)
    pT = (px**2 + py**2)**0.5
    try:
        eta = torch.arcsinh(pz / (pT + eps))
    except AttributeError:
        eta = arcsinh(pz / (pT + eps))
    phi = torch.atan2(py+eps, px+eps)
    return torch.stack((E, pT, eta, phi), dim=-1)


def get_jet_p4(p: torch.Tensor) -> torch.Tensor:
    return torch.sum(p, dim=-2)


def get_polar_rel(
    p: torch.Tensor,
    jet_p: torch.Tensor,
    eps: float = EPS_DEFAULT
) -> torch.Tensor:
    """Get polar coordinates relative to the jet.

    :param p: Particle features in (pt, eta, phi) or (E, pt_eta, phi).
    :type p: torch.Tensor
    :param jet_p: Jet features in (Pt, Eta, Phi) or (EJet, Pt, Eta, Phi).
    :type jet_p: torch.Tensor
    :return: Polar coordinates relative to the jet (pt_rel, eta_rel, phi_re;).
    :rtype: torch.Tensor
    """
    if p.shape[-1] == 4:
        _, pt, eta, phi = p.unbind(-1)
    elif p.shape[-1] == 3:
        pt, eta, phi = p.unbind(-1)
    else:
        raise ValueError(
            f"Invalid shape for p: {p.shape}. "
            "Feature dimension must be 3 or 4."
        )

    if jet_p.shape[-1] == 4:
        _, jet_pt, jet_eta, jet_phi = jet_p.unbind(-1)
    elif jet_p.shape[-1] == 3:
        jet_pt, jet_eta, jet_phi = jet_p.unbind(-1)
    else:
        raise ValueError(
            f"Invalid shape for jet_p: {jet_p.shape}. "
            "Feature dimension must be 3 or 4."
        )

    pt_norm = pt / (jet_pt.unsqueeze(-1) + eps)
    eta_norm = eta - jet_eta.unsqueeze(-1)
    phi_norm = phi - jet_phi.unsqueeze(-1)
    phi_norm = (phi_norm + np.pi) % (2 * np.pi) - \
        np.pi  # normalize to [-pi, pi)
    return torch.stack((pt_norm, eta_norm, phi_norm), dim=-1)

class DistanceDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__()
        if x.shape != y.shape:
            raise ValueError(
                f"x and y shapes do not match: "
                f"{x.shape} != {y.shape}"
            )

        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self.x[idx], self.y[idx])
