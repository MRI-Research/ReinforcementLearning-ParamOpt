import torch
from skimage.filters import sobel
import numpy as np

def rss_ifft_all_slices(kspace: torch.Tensor) -> torch.Tensor:
    """
    2-D centered iFFT + RSS combine, returns [S, H, W]
    """
    # kspace: [S, C, H, W] 
    x = torch.fft.ifftshift(kspace, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    mag = x.abs()                         
    return torch.sqrt((mag**2).sum(dim=1))  



def compute_edge_noise_reward(sim_imgs: torch.Tensor) -> float:
        """
        Compute a segmentation-free reward that peaks at an interior TE.
        1) Use Sobel to get per-slice gradient magnitude.
        2) For each slice, take only the top 10% of gradient magnitudes
           (to ignore noisy gradients) and average them.
        3) Estimate noise as the global std of the volume.
        Reward = edge_strength / (noise + eps).
        """
        vol = sim_imgs.detach().cpu().numpy()   
        edge_scores = []
        for sl in vol:
            grad = sobel(sl.astype(np.float32)) 
            # pick top 10% of gradient values
            thresh = np.percentile(grad, 90)
            strong = grad[grad > thresh]
            if strong.size:
                edge_scores.append(strong.mean())
        if not edge_scores:
            return 0.0
        edge_strength = float(np.mean(edge_scores))

        # noise estimate = std of entire volume
        noise = float(vol.std())

        return edge_strength / (noise + 1e-6)
    
def compute_image_score(ref: torch.Tensor, sim_imgs: torch.Tensor) -> float:
    """
    Compute negative MSE with shared normalization for reward.
    """
    ref_np  = ref.cpu().numpy()
    sim_np  = sim_imgs.cpu().numpy()
    # Shared normalization using reference image's range
    dr       = float(ref_np.max() - ref_np.min()) if ref_np.max() != ref_np.min() else 1.0
    ref_norm = (ref_np - ref_np.min()) / dr
    sim_norm = (sim_np - ref_np.min()) / dr
    mse      = float(((ref_norm - sim_norm) ** 2).mean())
    return -mse  # Negative MSE to maximize similarity


def parameter_stability_score(tr,tr_min,te,te_min,tr_max,te_max,device):
    """
    Compute a stability score based on the difference between the current and target TR/TE values.
    The score is normalized to be between 0 and 1.
    """
    tr_mid = 0.5 * (tr_min + tr_max)
    te_mid = 0.5 * (te_min + te_max)
    tr_tensor = torch.tensor(tr, device=device)
    te_tensor = torch.tensor(te, device=device)
    stab_tr = 1.0 - torch.abs(tr_tensor - tr_mid)/(tr_max - tr_min)
    stab_te = 1.0 - torch.abs(te_tensor - te_mid)/(te_max - te_min)
    stability = (stab_tr + stab_te)/2.0
        
    return stability

