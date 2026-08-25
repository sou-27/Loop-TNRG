# Loop-TNRG

Loop-optimized tensor network renormalization group algorithm implemented in python. The hamiltonian used can be found in : arXiv:2604.20201.
The algorithm is described in : Bao, Chenfeng. "Loop optimization of tensor network renormalization: algorithms and applications." (2019), Yang, Shuo, Zheng-Cheng Gu, and Xiao-Gang Wen. "Loop optimization for tensor network renormalization." Physical review letters 118.11 (2017): 110504.

## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:YOUR_USERNAME/loop_tnrg.git
   cd loop_tnrg

2. Create a virtual environment and install the package in editable mode:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .

## Usage

Run with the parameters r,theta that specify the hamiltonian (both float) and chi (int) specfies maximum bond dimension to be kept during RG.

```bash
python scripts/do_rg.py r theta chi

Optional arguments include
--n_layers : number of layers of tensor network to be comressed during preconditioning. default = 6
--niterations : number of RG iterations. default = 12
--niter : number of passes in loop-optimization step. default = 30
--chi_init : maximum bond dimension to initialize tensor network in. default = 12
--output_dir : directory to save outputs. defailt = "results"

