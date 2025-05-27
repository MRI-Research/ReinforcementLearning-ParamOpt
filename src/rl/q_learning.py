import numpy as np
import json
import torch

def q_learn(env, episodes=250,  device='cpu'):
    alpha=0.1 
    gamma=0.95
    eps_start=1.0 
    eps_end=0.1 
    eps_decay=0.9995
    
    Q = {}
    eps = eps_start
    rewards = []
    device = torch.device(device)
    
    for ep in range(episodes):
        s = env.reset()
        total_r = 0
        kk=500
        for it in range(kk):
            if s not in Q: 
                Q[s] = torch.zeros(len(env.actions), device=device, dtype=torch.float32)
            
            if torch.rand(1, device=device).item() < eps: 
                a = torch.randint(0, len(env.actions), (1,), device=device).item()
            else:
                a = int(torch.argmax(Q[s]).item())

            next_s, r, done = env.step(a)
            total_r += r
            
            print(f"Episode {ep+1}/{episodes}, Iteration {it+1}/{kk}")
            if next_s not in Q:
                Q[next_s] = torch.zeros(len(env.actions), device=device, dtype=torch.float32)
            
            target = r + gamma * torch.max(Q[next_s]).item()
            Q[s][a] += alpha * (target - Q[s][a].item())

            s = next_s
            if done:
                break

            eps = max(eps_end, eps * eps_decay)
        rewards.append(total_r)
            
    return Q, rewards



