import torch
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


def compute_image_score(ref, sim_imgs):
        """Compute negative MSE with shared normalization for reward."""
        ref_np = ref.cpu().numpy()
        sim_np = sim_imgs.cpu().numpy()
        dr = float(ref_np.max() - ref_np.min()) if ref_np.max() != ref_np.min() else 1.0
        ref_norm = (ref_np - ref_np.min()) / dr
        sim_norm = (sim_np - ref_np.min()) / dr
        
        mse = ((ref_norm - sim_norm) ** 2).mean()
        return -mse  # Negative MSE 

def compute_parameter_optimality_reward(tr_min,tr_max,te_min,te_max, tr, te):
        """
        Combined reward for TR and TE stability.
        """
        # TR optimization
        optimal_tr_min = tr_min  
        optimal_tr_max = tr_max  
        
        if optimal_tr_min <= tr <= optimal_tr_max:
            tr_reward = 1.0
        else:
            if tr < optimal_tr_min:
                distance = np.abs(optimal_tr_min - tr)
                tr_penalty = -distance 
            else:
                distance = np.abs(tr - optimal_tr_max)
                tr_penalty = -distance 
            tr_reward = tr_penalty
        
        # TE optimization  
        optimal_te_min = te_min  
        optimal_te_max = te_max   
        
        if optimal_te_min <= te <= optimal_te_max:
            te_reward = 1.0
        else:
            if te < optimal_te_min:
                distance = np.abs(optimal_te_min - te)
                te_penalty = -distance 
            else:
                distance = np.abs(te - optimal_te_max)
                te_penalty = -distance 
            te_reward = te_penalty       
       
        combined_reward = (tr_reward + te_reward) / 2.0
        
        
        if tr_reward > 0 and te_reward > 0:
            combined_reward += 0.5  # Bonus for having both parameters optimal
        
        return combined_reward
