import numpy as np
import torch
from utils.image_processing import compute_image_score, compute_parameter_optimality_reward
from collections import deque


class MRIEnv:
    """
    Discrete Q-learning environment for MRI parameter optimization.
    - state = (tr, te) in ms
    - action = (dTR, dTE) in ms
    - reward = mixed score based on image similarity, diversity, and parameter optimality
    """
    
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

        tr_total_range = self.tr_max - self.tr_min 
        te_total_range = self.te_max - self.te_min 
        step_size=0.01
        c1=  step_size * tr_total_range
        c2=  step_size * te_total_range

        self.actions = [(dTR*c1, dTE*c2) for dTR in (-1, 0, 1) for dTE in (-1, 0, 1)]

        # For reward shaping
        self.best_reward = float('-inf')
        self.last_rewards    = deque(maxlen=10) # last 10 rewards
        self.reward_history = []
        self.episode_step = 0
        self.stagnation_count = 0
        # For diversity bonus
        self.visited_states = {}

    def compute_diversity_bonus(self, state):
        """Encourage exploration of unvisited or rarely visited states"""
        state_key = (round(state[0]), round(state[1]))
        visit_count = self.visited_states.get(state_key, 0)
        self.visited_states[state_key] = visit_count + 1
        
        # Higher bonus for less visited states
        diversity_bonus = 1.0 / (1.0 + visit_count)
        return diversity_bonus
    

    def compute_progress_bonus(self, base_reward):
        """Reward improvement over recent performance"""
        self.reward_history.append(base_reward)
        
        # Keep only recent history
        if len(self.reward_history) > 10:
            self.reward_history.pop(0)
        
        if len(self.reward_history) < 3:
            return 0.0
        
        # Compare current reward to recent average
        recent_avg = np.mean(self.reward_history[:-1])
        progress = base_reward - recent_avg
        # Bonus for improvement, penalty for stagnation
        progress_bonus = np.tanh(progress * 10)  # clipped to [-1, 1]
        return progress_bonus
      
    def _reward(self, sim_imgs: torch.Tensor) -> float:
        """Enhanced reward function with multiple components"""
        tr, te = self.state
        a, b, c = 5, 5, 1000.0
        
        base_ssim = compute_image_score(self.ref, sim_imgs)
        diversity_bonus = self.compute_diversity_bonus(self.state)
        t_optimality = compute_parameter_optimality_reward(self.tr_min,self.tr_max,self.te_min,self.te_max,tr, te)
        progress_bonus = self.compute_progress_bonus(c * base_ssim)
         
        # Combine all components
        total_reward = (
            c * base_ssim +
            a * diversity_bonus +  # Strong early exploration
            t_optimality +  # Domain knowledge
            b * progress_bonus  # Reward improvement
        )
        
        self.last_rewards.append(total_reward)
        #mean_last_10 = sum(self.last_rewards) / len(self.last_rewards) # compute mean of whatever we have so far (up to 10) (Optional)
        
        # Track best reward for potential reward shaping
        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        components = {
            'image_score': c * base_ssim,
            'diversity_bonus': a * diversity_bonus,
            'te_optimality':  t_optimality,
            'progress_bonus': b * progress_bonus,
        }
        
        return total_reward, components
    
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
            - reward: computed mixed reward
            - done: boolean indicating if the episode is done'''

        old_tr, old_te = self.state
        dTR, dTE = self.actions[action_idx]
        tr_, te_ = old_tr + dTR, old_te + dTE
        
        outside = False
        if tr_ < self.tr_min: tr_, outside = self.tr_min, True
        if tr_ > self.tr_max: tr_, outside = self.tr_max, True
        if te_ < self.te_min: te_, outside = self.te_min, True
        if te_ > self.te_max: te_, outside = self.te_max, True
        
        sim_img = self._simulate(tr_, te_)
        reward, _ = self._reward(sim_img)
        
        if outside:
            reward *= 0.5  
        
        if self.stagnation_count > 10:
            reward -= 5.0  # Encourage trying new actions
        
        done = (tr_ == self.tr0 and te_ == self.te0)
        self.state = (tr_, te_)
        self.episode_step += 1
        
        return self.state, reward, done
    
    def reset(self):
        tr = torch.randint(int(self.tr_min), int(self.tr_max) + 1, (1,), device=self.device).item()
        te = torch.randint(int(self.te_min), int(self.te_max) + 1, (1,), device=self.device).item()
        self.state = (float(tr), float(te))
        
        # Reset episode-specific tracking
        self.episode_step = 0
        self.stagnation_count = 0
        self.reward_history = []
        
        return self.state
        
