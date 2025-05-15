import numpy as np

def q_learn(env, episodes=200, alpha=0.1, gamma=0.95,
            eps_start=1.0, eps_end=0.1, eps_decay=0.99):
    Q = {}
    eps = eps_start
    rewards = []
    for ep in range(episodes):
        s = env.reset()
        total_r = 0
        for _ in range(1500): 
            if s not in Q: 
                Q[s] = np.zeros(len(env.actions))
            if np.random.rand() < eps:
                a = np.random.randint(len(env.actions))
            else:
                a = int(np.argmax(Q[s]))

            next_s, r, done = env.step(a)
            total_r += r

            if next_s not in Q:
                Q[next_s] = np.zeros(len(env.actions), dtype=np.float32)
            Q[s][a] += alpha * (r + gamma * Q[next_s].max() - Q[s][a])

            s = next_s
            if done:
                break

            eps = max(eps_end, eps * eps_decay)
        rewards.append(total_r)
            
    return Q, rewards