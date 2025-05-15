## Info

This project implements a RL approach to optimize MRI parameters using Q-learning. The goal is to recover the optimal parameters that yield the best image quality based on the Structural Similarity Index (SSIM).

## Features
- **Custom Reinforcement Learning Environment**: Simulates MRI parameter optimization using the `MRIEnv` class benefiting from spin-echo equations.
- **Q-Learning Implementation**: Learns the optimal policy for parameter optimization.
- **SSIM-Based Rewards**: Evaluates image quality using the SSIM.
- **Modular Design**: Organized into reusable modules for better maintainability, current version is for [FastMRI Knee](https://fastmri.med.nyu.edu/).

## Getting Started
```
# Clone the repository
$ git clone https://github.com/your‑username/your‑repo.git
$ cd your‑repo

# (Optional) create a virtual environment
$ python -m venv .venv
$ source .venv/bin/activate  # For Linux


# Install dependencies
$ pip install -r requirements.txt

# Run the application
$ python src/main.py
```


## Directory  Structure
```
src/
├── main.py            # Entry point for the application
├── rl/
│   ├── environment.py # Defines the MRIEnv class
│   └── q_learning.py  # Implements the Q‑learning algorithm
└── utils/
    ├── image_processing.py # Image processing utilities
    └── xml_parsing.py      # XML header parsing utilities
```
