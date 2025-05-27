import numpy as np
import torch
from utils.image_processing import compute_edge_noise_reward, compute_image_score, parameter_stability_score



class MRIEnv:
    """
    Discrete Q-learning environment for MRI parameter optimization.
    - state = (tr, te) in ms
    - action = (dTR, dTE) in ms
    - reward = mixed score based on edge noise, stability, and image similarity
    """
    actions = [(dTR*10, dTE) for dTR in (-1, 0, 1) for dTE in (-1, 0, 1)]

    def __init__(self, ref_img, tr0, te0, T1c, T2c,device):
        self.device=torch.device(device)
        if isinstance(ref_img, np.ndarray):
            self.ref = torch.from_numpy(ref_img).float().to(self.device)
        else:
            self.ref = ref_img.to(self.device)
        if isinstance(T1c, torch.Tensor):
            self.T1c = T1c.to(self.device)
        else:
            self.T1c = torch.tensor(T1c, device=self.device, dtype=torch.float32)
            
        if isinstance(T2c, torch.Tensor):
            self.T2c = T2c.to(self.device)
        else:
            self.T2c = torch.tensor(T2c, device=self.device, dtype=torch.float32)
        self.tr0 = tr0
        self.te0 = te0
        self.tr_min, self.tr_max = tr0 * 0.5, tr0 * 2
        self.te_min, self.te_max = te0 * 0.5, te0 * 2
        Rf0 = 1 - torch.exp(-torch.tensor(tr0, device=self.device) / self.T1c)
        Ef0 = torch.exp(-torch.tensor(te0, device=self.device) / self.T2c)
        self.M0 = self.ref / (Rf0 * Ef0)
      
    def _reward(self, sim_imgs: torch.Tensor) -> float:
        '''Compute the reward based on simulated images.'''
        tr, te = self.state
        tr_min, te_min = self.tr_min, self.te_min
        tr_max, te_max = self.tr_max, self.te_max
        device = self.device
        
        enr = compute_edge_noise_reward(sim_imgs)
        st=parameter_stability_score(tr,tr_min,te,te_min,tr_max,te_max,device)
        base_ssim = compute_image_score(self.ref, sim_imgs)
        
        a, b, c = 1.0, 0.1, 1000.0
        total = a * enr + b * st + c* base_ssim 

        return total, {'edge_noise': a*enr, 'stability': b*st, 'image_score': c*base_ssim}
    
    def _simulate(self, tr, te) -> torch.Tensor:
        '''Simulate MRI image based on TR and TE.'''
        # https://www.cis.rit.edu/htbooks/mri/chap-4/chap-4-h5.htm
        tr_tensor = torch.tensor(tr, device=self.device, dtype=torch.float32)
        te_tensor = torch.tensor(te, device=self.device, dtype=torch.float32)
        
        Rf = 1 - torch.exp(-tr_tensor / self.T1c)
        Ef = torch.exp(-te_tensor / self.T2c)
        sim_img = self.M0 * Rf * Ef
        noise = torch.randn_like(sim_img, device=self.device) * 0.01 * sim_img.max()
        return sim_img + noise

    def step(self, action_idx):
        '''Perform a step in the environment.
        action_idx: index of the action to take (0-8)
        Returns:
            - state: new state (tr_, te_)
            - reward: computed reward
            - done: boolean indicating if the episode is done'''

        old_tr, old_te = self.state
        dTR, dTE       = self.actions[action_idx]
        tr_, te_       = old_tr + dTR, old_te + dTE

        outside = False
        if tr_ < self.tr_min: tr_, outside = self.tr_min, True
        if tr_ > self.tr_max: tr_, outside = self.tr_max, True
        if te_ < self.te_min: te_, outside = self.te_min, True
        if te_ > self.te_max: te_, outside = self.te_max, True
        sim_img = self._simulate(tr_, te_)
        reward, comps = self._reward(sim_img)
        if outside:
            reward *= 0.5
        done = (tr_ == self.tr0 and te_ == self.te0)
        self.state = (tr_, te_)
        
        return self.state, reward, done
    
    def reset(self):
        tr = torch.randint(int(self.tr_min), int(self.tr_max) + 1, (1,), device=self.device).item()
        te = torch.randint(int(self.te_min), int(self.te_max) + 1, (1,), device=self.device).item()
        self.state = (float(tr), float(te))
        
        return self.state
        
