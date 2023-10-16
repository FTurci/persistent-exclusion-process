#!/usr/bin/env python
"""Evolve lattices and save the dataset for training

Usage:
-----
./sampler.py

"""

import h5py
import numpy as np
import tqdm

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
    n_x = n_y = 128
    n_p = int(0.05 * n_x * n_y)
    for tumble in tqdm.tqdm(np.logspace(-3, -1, 3, base=2)):
        print("Tumble", tumble)
        snapshot = int(1 / tumble)
        speed = 10
        n_iter = int(1000 * snapshot)
        print("Total number of iterations is", n_iter)
        lat = lattice.Lattice(n_x * n_y, n_p)
        lat.set_square_connectivity(n_x, n_y)
        lat.reset_random_occupancy()
        lat.reset_orientations()
        # NOTE: warmup lattice
        for _ in range(500):
            lat.c_move(tumble, speed)
        with h5py.File(
            f"../data/dataset_tumble_{np.log10(tumble):.4f}.h5", "w"
        ) as f_out:
            for iteration in tqdm.tqdm(range(n_iter)):
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
