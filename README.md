## Info

This project implements a RL approach to optimize MRI parameters using Q-learning. The goal is to recover the optimal parameters based on image quality and parameter stability.

## Features
- **Custom Reinforcement Learning Environment**: Simulates MRI parameter optimization using the `MRIEnv` class benefiting from spin-echo equations.
- **Q-Learning Implementation**: Learns the optimal policy for parameter optimization.
- **Image Processing:** Converts to images with an inverse FFT and root-sum-of-squares reconstruction.
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
<pre>
src/
├── <a href="https://github.com/MRI-Research/ReinforcementLearning-ParamOpt/blob/main/src/main.py">main.py</a>          # Entry point for the application
├── rl/
│   ├── <a href="https://github.com/MRI-Research/ReinforcementLearning-ParamOpt/blob/main/src/rl/environment.py">environment.py</a>  # Defines the MRIEnv class
│   └── <a href="https://github.com/MRI-Research/ReinforcementLearning-ParamOpt/blob/main/src/rl/q_learning.py">q_learning.py</a>   # Implements the Q-learning algorithm
└── utils/
    ├── <a href="https://github.com/MRI-Research/ReinforcementLearning-ParamOpt/blob/main/src/utils/image_processing.py">image_processing.py</a> # Image processing utilities
    └── <a href="https://github.com/MRI-Research/ReinforcementLearning-ParamOpt/blob/main/src/utils/xml_parsing.py">xml_parsing.py</a>      # XML header parsing utilities
</pre>
