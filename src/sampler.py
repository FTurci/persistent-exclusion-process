#!/usr/bin/env python
"""Evolve lattices and save the dataset for training

Usage:
-----
./sampler.py -h

"""

import h5py
import numpy as np
import tqdm
import pandas as pd

import argparse
import os

import lattice


def main():
    """Create dataset for model training

    Explanation:
    -----------
    - Create an even logspace (base 2) for tumble probability (?)
    - Create a lattice and warm it up (evolve 500 steps)
    - Create a dataset at each iteration (except condition below) (why ?)
    - Create also a dataset for each "image", but rolled-over (from 0 to 10) (why?)

    """
    parser = argparse.ArgumentParser(description="Generate some datasets")
    parser.add_argument("--density", help="Lattice density", type=float, required=True)
    parser.add_argument("--odd", help="Run odd indices of whole logspace (default: False)", action="store_true", default=False)
    args = parser.parse_args()
    
    speed = 10
    n_x = n_y = 128
    n_p = int(args.density * n_x * n_y)
    
    start = 1 if args.odd else 0
    tumbles = np.logspace(-6, -1, 10, base=2)[start::2]
    iters = 1000 * (1 / tumbles).astype(int)
    
    
    df = pd.DataFrame()
    df["tumble"] = tumbles
    df["n_iter"] = iters
    df["density"] = np.full_like(tumbles, args.density)
    df["speed"] = np.full_like(tumbles, speed, dtype=np.int_)
    if not os.path.isfile("../data/sampler_records.csv"):
        os.system("echo ',tumble,n_iter,density,speed' > ../data/sampler_records.csv")
    df.to_csv("../data/sampler_records.csv", mode="a", header=False)
    
    for idx, tumble in tqdm.tqdm(enumerate(tumbles)):
        print("Tumble", tumble)
        snapshot = int(1 / tumble)
        print("Total number of iterations is", iters[idx])
        lat = lattice.Lattice(n_x * n_y, n_p)
        lat.set_square_connectivity(n_x, n_y)
        lat.reset_random_occupancy()
        lat.reset_orientations()
        # NOTE: warmup lattice
        for _ in range(500):
            lat.c_move(tumble, speed)
        with h5py.File(
            f"../data/dataset_tumble_{tumble:.3f}_{args.density}.h5", "w"

        ) as f_out:
            for iteration in tqdm.tqdm(range(iters[idx])):
                lat.c_move(tumble, speed)
                if (iteration % snapshot) != 0:
                    continue
                f_out.create_dataset(
                    f"conf_{iteration}", data=lat.image().astype(np.int32)
                )
                for roll in range(0, n_x, 10):
                    f_out.create_dataset(
                        f"conf_{iteration}_{roll}",
                        data=np.roll(
                            lat.image().astype(np.uint8), (roll, roll), axis=(0, 1)
                        ),
                    )

if __name__ == "__main__":
    main()
