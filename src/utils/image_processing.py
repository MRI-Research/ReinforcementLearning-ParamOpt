import torch
from skimage.metrics import structural_similarity as ssim


def rss_ifft_torch(kspace: torch.Tensor) -> torch.Tensor:
    """2-D centered iFFT + RSS combine"""
    if kspace.ndim == 4:              
        kspace = kspace[0] # drop batch dim
    # IFFT
    x = torch.fft.ifftshift(kspace, dim=(-2, -1))
    x = torch.fft.ifft2(x, norm='ortho')
    x = torch.fft.fftshift(x, dim=(-2, -1))
    mag = x.abs()
    return torch.sqrt(torch.sum(mag**2, dim=0))  


def compute_ssim(ref_img: torch.Tensor, sim_img: torch.Tensor) -> float:
    """Compute SSIM between reference and simulated images."""
    ref = (ref_img - ref_img.min()) / (ref_img.max() - ref_img.min())
    model = (sim_img - sim_img.min()) / (sim_img.max() - sim_img.min())
    return ssim(ref.numpy(), model.numpy(), data_range=1.0)