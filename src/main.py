from rl.environment import MRIEnv
from rl.q_learning import q_learn
from utils.image_processing import rss_ifft_all_slices
from utils.xml_parsing import parse_header
from utils.log_rl import setup_logger
import h5py
import torch
import numpy as np
from pathlib import Path

def main(use_gpu):
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    root_dir = Path("")  # Path to FastMRI knee dataset
    file_list = sorted([f for f in root_dir.glob("*.h5")])

    logger = setup_logger('rl_knee_parameters.log')
    logger.info("Starting Q-learning runs on FastMRI knee dataset")
    results, all_rewards, file_it = [], [], 0
    for fpath in file_list:
        file_it+=1
        with h5py.File(fpath, 'r') as hf:
            ksp = torch.from_numpy(hf['kspace'][()]).to(device)
            # tr: repetition time
            # te: echo time
            # T1c: T1 relaxation time (the tissue’s longitudinal relaxation time)
            # T2c: T2 relaxation time (the tissue’s transverse relaxation time)
            tr0, te0, T1c, T2c = parse_header(hf['ismrmrd_header'][()])
            logger.debug("Original Values: TR0=%f, TE0=%f, T1c=%f, T2c=%f", tr0, te0, T1c, T2c)
            print(f"file_it/total file: {file_it}/{len(file_list)}")
            
            ref_imgs = rss_ifft_all_slices(ksp)
            env = MRIEnv(ref_imgs, tr0, te0, T1c, T2c,device)
            Q, rews = q_learn(env, episodes=10, steps=100, device=device)
            all_rewards.append(rews)
            logger.info("Completed Q-learning: max reward=%.4f", max(rews))

            s = env.reset()
            for _ in range(100): 
                q_values = Q.get(s, torch.zeros(len(env.actions), device=device))
                a = int(torch.argmax(q_values).item())
                s, _, done = env.step(a)
                if done:
                    break
            results.append(((tr0, te0), s))

        # log when found
        real_p = (tr0, te0)
        found_p = s
        msg = f"File {fpath}: Real {real_p} vs Found {found_p}"
        print(msg)
        logger.info(msg)
    
    # log general comparison
    for i, (real_p, found_p) in enumerate(results, 1):
        print(f"File {i}: Real {real_p} vs Found {found_p}")
        msg = f"File {i}: Real {real_p} vs Found {found_p}"
        logger.info(msg)

if __name__ == "__main__":
    main(use_gpu=True)