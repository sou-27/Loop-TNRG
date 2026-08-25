import numpy as np
import argparse
from loop_tnrg.rg import reduced_tnrg
import h5py
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description= "Store (r,theta,chi)")

    parser.add_argument('r', type=float)
    parser.add_argument('theta', type = float)
    parser.add_argument('chi', type = int)
    parser.add_argument('--n_layers', type = int, default = 6)
    parser.add_argument('--niterations', type = int, default = 12)
    parser.add_argument('--niter', type = int, default = 30)
    parser.add_argument('--chi_init', type = int, default = 20)
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output files")


    args = parser.parse_args()

    ouptut_dir = Path(args.output_dir)
    ouptut_dir.mkdir(parents=True, exist_ok=True)


    r = float(args.r)
    th = np.pi * float(args.theta)
    chi = int(args.chi)
    n_layers = int(args.n_layers)
    delta_t = 2**(-n_layers)
    niterations = int(args.niterations)
    niter = int(args.niter)
    repel = 30
    chi_init = int(args.chi_init)

    print(f"Starting LoopTNRG for (r,theta,chi = ({r}, {th}, {chi}))")
    CC,SD = reduced_tnrg(r,th,repel, delta_t, chi_init, n_layers, chi, niterations, niter)

    save_path = ouptut_dir / "rg_output.h5"

    with h5py.File(save_path, "w") as f:
        f.create_dataset("central_charge", data = CC, compression="gzip")
        f.create_dataset("scaling_dimensions", data = SD, compression="gzip")

        f.attrs["r"] = r
        f.attrs["theta"] = th
        f.attrs["chi"] = chi

if __name__ == "__main__":
    main()


    