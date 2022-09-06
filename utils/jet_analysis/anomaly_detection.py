from sklearn import metrics
from typing import Callable, Dict, Optional, Tuple, Union
from matplotlib import pyplot as plt
from .utils import arcsinh
from pathlib import Path

import torch
import energyflow
from tqdm import tqdm
import numpy as np

EPS_DEFAULT = 1e-16
# keys for scores
PARTICLE_CARTESIAN = "particle, Cartesian"
PARTICLE_POLAR = "particle, polar"
PARTICLE_NORMALIZED_CARTESIAN = "particle, normalized Cartesian"
PARTICLE_NORMALIZED_POLAR = "particle, normalized polar"
PARTICLE_RELATIVE_POLAR = "particle, relative polar"
JET_CARTESIAN = "jet, Cartesian"
JET_POLAR = "jet, polar"
PARTICLE_LORENTZ = "particle, Lorentz norms"
JET_LORENTZ = "jet, Lorentz norms"
EMD = 'emd'
EMD_RELATIVE = 'emd (relative coordinates)'

def anomaly_detection_ROC_AUC(
    sig_recons: torch.Tensor, 
    sig_target: torch.Tensor,
    sig_recons_normalized: torch.Tensor, 
    sig_target_normalized: torch.Tensor,
    bkg_recons: torch.Tensor, 
    bkg_target: torch.Tensor,
    bkg_recons_normalized: torch.Tensor, 
    bkg_target_normalized: torch.Tensor,
    include_emd: bool = True,
    save_path: Union[str, Path] = None,
    plot_rocs: bool = True,
    plot_num_rocs: Optional[int] = -1
) -> Tuple[Dict[str, Tuple[np.ndarray]], Dict[str, Tuple[np.ndarray]]]:
    """Compute get AUC and ROC curves in the anomaly detection.

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
    :param save_path: Path to save the ROC curves and AUCs, defaults to None. 
    If None, the ROC curves and AUCs are not saved.
    :type save_path: str, optional
    :param plot_num_rocs: Number of ROC curves to keep when plotting 
    (after sorted by AUC), defaults to -1.
    If the value takes one of {None, 0, -1}, all ROC curves are kept.
    :type plot_num_rocs: int, optional
    :return: (`roc_curves`, `aucs`), 
    where `roc_curves` is a dictionary {kind: roc_curve}, 
    and `aucs` is a dictionary {kind: auc}.
    :rtype: Tuple[Dict[str, Tuple[np.ndarray]], Dict[str, Tuple[np.ndarray]]]
    """
    scores_dict, true_labels = anomaly_scores_sig_bkg(
        sig_recons, sig_target, sig_recons_normalized, sig_target_normalized,
        bkg_recons, bkg_target, bkg_recons_normalized, bkg_target_normalized,
        include_emd=include_emd
    )
    roc_curves = dict()
    aucs = dict()
    for kind, scores in scores_dict.items():
        roc_curve = metrics.roc_curve(true_labels, scores)
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
        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.set_xlabel('True Positive Rate')
        ax.set_ylabel('False Positive Rate')
        ax.set_yscale('log')
        
        auc_sorted = sorted(aucs.items(), key=lambda x: x[1], reverse=True)
        for i, (kind, auc) in enumerate(auc_sorted):
            if (plot_num_rocs is not None) and (plot_num_rocs > 0):
                if i >= plot_num_rocs:
                    break
                
            fpr, tpr, thresholds = roc_curves[kind]
            ax.plot(tpr, fpr, label=f'{kind} (AUC: {auc:.5f})')
            ax.plot(
                np.linspace(0, 1, 100), [0.01] * 100,
                '--', c='gray', linewidth=1 
            )
            ax.vlines(
                x=tpr[np.searchsorted(fpr, 0.01)],
                ymin=0, ymax=0.01,
                linestyles="--", colors="gray", linewidth=1
            )
            
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path / 'roc_curves.pdf')
        else:
            plt.show()

    return roc_curves, aucs

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
    :return: (true_labels, scores), where scores is a dictionary 
    with the scores (value) for each type (key).
    :rtype: Tuple[np.ndarray, Dict[str, np.ndarray]]
    """
    sig_scores = anomaly_scores(
        sig_recons, 
        sig_target,
        sig_recons_normalized, 
        sig_target_normalized,
        include_emd=include_emd
    )
    bkg_scores = anomaly_scores(
        bkg_recons, 
        bkg_target,
        bkg_recons_normalized, 
        bkg_target_normalized,
        include_emd=include_emd
    )
    scores = {
        k: np.concatenate([sig_scores[k], bkg_scores[k]])
        for k in sig_scores.keys()
    }
    true_labels = np.concatenate([
        np.ones_like(sig_scores[PARTICLE_CARTESIAN]),
        -np.ones_like(bkg_scores[PARTICLE_CARTESIAN])
    ])
    return scores, true_labels

def anomaly_scores(
    recons: torch.Tensor, 
    target: torch.Tensor,
    recons_normalized: torch.Tensor, 
    target_normalized: torch.Tensor,
    include_emd: bool = True,
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
        PARTICLE_CARTESIAN: mse(recons, target).mean(-1).numpy(),  # average over jets
        PARTICLE_POLAR: mse(recons_polar, target_polar).mean(-1).numpy(),
        PARTICLE_NORMALIZED_CARTESIAN: mse(recons_normalized, target_normalized).mean(-1).numpy(),
        PARTICLE_NORMALIZED_POLAR: mse(recons_normalized_polar, target_normalized_polar).mean(-1).numpy(),
        PARTICLE_RELATIVE_POLAR: mse(recons_polar_rel, target_polar_rel).mean(-1).numpy(),
        JET_CARTESIAN: mse(recons_jet, target_jet).numpy(),
        JET_POLAR: mse(recons_jet, target_jet).numpy(), 
        PARTICLE_LORENTZ: mse_lorentz(recons, target).mean(-1).numpy(),
        JET_LORENTZ: mse_lorentz(recons_jet, target_jet).numpy()
    }
    
    if include_emd:
        scores[EMD] = emd_loss(recons_polar, target_polar)
        scores[EMD_RELATIVE] = emd_loss(target_polar_rel, recons_polar_rel)
    
    return scores


# Helper functions
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
    def norm_sq_Lorentz(x: torch.Tensor) -> torch.Tensor:
        E, px, py, pz = x.unbind(-1)
        return E**2 - px**2 - py**2 - pz**2
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
    phi_norm = (phi_norm + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi)
    return torch.stack((pt_norm, eta_norm, phi_norm), dim=-1)