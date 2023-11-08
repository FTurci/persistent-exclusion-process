#!/usr/bin/env python
"""
Display lattice using pyplot

Usage:
-----
./view.py

"""
import argparse
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from scipy import ndimage

from plot_utils import get_plot_configs
from stringato import extract_floats
from utils import get_ds_iters


def main():
    """Make a grid of snapshops of the last iteration"""
    parser = argparse.ArgumentParser(description="Generate some datasets")
    parser.add_argument(
        "--csize",
        help="Plot cluster size distribution instead",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()
    plot_configs = get_plot_configs()
    plot_configs["xtick.labelsize"] = 8
    plot_configs["ytick.labelsize"] = 8
    plt.rcParams.update(plot_configs)
    fig = plt.figure(figsize=(9 * 3 / 5, 9), constrained_layout=True)
    gspec = fig.add_gridspec(5, 3, wspace=0.15, hspace=0.15)
    cmap = plt.get_cmap(name="cmc.bilbaoS", lut=5)
    files = glob.glob("../data/dataset*")
    stuff = []
    for file in files:
        tumble = float(extract_floats(file)[0])
        density = float(extract_floats(file)[1])
        stuff.append((tumble, density))
    files = []
    for idx, pair in enumerate(sorted(stuff)[::2]):
        if idx % 5 not in (0, 2, 4):
            continue
        files.append(f"../data/dataset_tumble_{pair[0]:.3f}_{pair[1]}.h5")
    ctr = 0
    text_kwrgs = {
        "bbox": {"boxstyle": "round", "facecolor": "white", "alpha": 0.5},
        "ha": "right",
    }
    for idx in range(5):
        for jdx in range(3):
            axis = fig.add_subplot(gspec[idx, jdx], autoscale_on=False)
            with h5py.File(files[ctr], "r") as fin:
                key_list = list(fin.keys())
                iter_n = get_ds_iters(key_list)
                img = fin[f"conf_{iter_n[-1]}"]
                text_kwrgs["s"] = r"$\alpha = {}, \phi = {}$".format(
                    files[ctr][25:30], files[ctr][31:35]
                )
                if args.csize:
                    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                    labelled, _ = ndimage.label(img, structure=kernel)
                    cluster_sizes = np.bincount(labelled.flatten())[1:]
                    min_c = cluster_sizes.min()
                    max_c = cluster_sizes.max()
                    bin_edges = np.linspace(min_c, max_c, 100)
                    counts, _ = np.histogram(
                        cluster_sizes, bins=bin_edges, density=True
                    )
                    axis.scatter(
                        bin_edges[:-1],
                        counts,
                        edgecolor=(0, 0, 0, 1),
                        facecolor=(0, 0, 0, 0.3),
                    )
                    axis.set_yscale("log"), axis.set_xscale("log")
                    axis.text(
                        y=0.89,
                        x=0.96,
                        transform=axis.transAxes,
                        **text_kwrgs,
                    )
                    fig.supxlabel("Cluster size")
                else:
                    axis.matshow(img, cmap=cmap)
                    axis.text(
                        y=-0.1,
                        x=0.96,
                        transform=axis.transAxes,
                        **text_kwrgs,
                    )
            axis.set_xlim((0, 120))
            axis.set_ylim((0, 120))
            if jdx != 0:
                axis.set_yticklabels([])
            if idx != 0:
                axis.set_xticklabels([])
            ctr += 1
    name = "csize" if args.csize else "default"
    fig.savefig(f"../plots/{name}.pdf")
    fig.savefig(f"../plots/{name}.png")


if __name__ == "__main__":
    main()
