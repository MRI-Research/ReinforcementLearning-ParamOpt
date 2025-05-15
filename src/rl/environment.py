import numpy as np
import torch
from utils.image_processing import compute_ssim

class MRIEnv:
    """
    Discrete Q-learning environment for MRI parameter optimization.
    - state = (TR, TE) in ms
    - action = (dTR, dTE) in ms
    - reward = SSIM-based
    """
    actions = [(dTR, dTE) for dTR in (-1, 0, 1) for dTE in (-1, 0, 1)]

    def __init__(self, ref_img, tr0, te0, T1c, T2c):
        self.ref = ref_img
        self.tr0 = tr0
        self.te0 = te0
        self.T1c = T1c
        self.T2c = T2c
        self.tr_min = tr0
        self.tr_max = tr0 * 1.5
        self.te_min = te0
        self.te_max = te0 * 1.5
        Rf0 = 1 - np.exp(-tr0 / T1c)
        Ef0 = np.exp(-te0 / T2c)
        self.M0 = self.ref / (Rf0 * Ef0) 

    def _reward(self, sim_img: torch.Tensor) -> float:
        """Compute SSIM(ref, sim)"""
        return compute_ssim(self.ref, sim_img)

    def reset(self):
        tr = np.random.randint(self.tr_min, self.tr_max + 1)
        te = np.random.randint(self.te_min, self.te_max + 1)
        self.state = (float(tr), float(te))
        return self.state

    def _simulate(self, tr, te) -> torch.Tensor:
        # https://www.cis.rit.edu/htbooks/mri/chap-4/chap-4-h5.htm
        Rf = 1 - np.exp(-tr / self.T1c)
        Ef = np.exp(-te / self.T2c)
        sim_img = self.M0 * Rf * Ef
        return sim_img

    def step(self, action_idx):
        dTR, dTE = self.actions[action_idx]
        tr, te = self.state
        tr_, te_ = tr + dTR, te + dTE
        outside = False
        if tr_ < self.tr_min: tr_, outside = self.tr_min, True
        if tr_ > self.tr_max: tr_, outside = self.tr_max, True
        if te_ < self.te_min: te_, outside = self.te_min, True
        if te_ > self.te_max: te_, outside = self.te_max, True
        sim_img = self._simulate(tr_, te_)
        reward = self._reward(sim_img)
        if outside:
            reward *= 0.5
        done = (tr_ == self.tr0 and te_ == self.te0)
        return self.state, reward, done