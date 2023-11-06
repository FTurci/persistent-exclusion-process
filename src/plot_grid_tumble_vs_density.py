#!/usr/bin/env python
"""
Display lattice using pyplot

Usage:
-----
./view.py

"""
import glob
import re

import h5py
import matplotlib.pyplot as plt

from plot_utils import get_plot_configs
from stringato import extract_floats


def get_ds_iters(key_list: list) -> list:
    """
    Get all the unique iteration numbers

    :param key_list: a list of all the possible dataset keys/names

    :returns: a list of unique iterations
    """
    iter_n = []
    for val in key_list:
        if re.search("^conf_\d+$", val):
            iter_n.append(int(val[5:]))
    return sorted(iter_n)


def main():
    """
    Go through all dataset files and update viewport iteratively

    Explanation
    -----------
    - For each file in the dataset, load in as hdf5 file (dict)
    - Tumbling information is at the top of the file, This goes into `outputs` (list)
      and is used as plot title
    - Value of each item in the hdf5 dict is parsed as 2d array into `matshow`
    - Only does it once (breaks when `count > 0`)

    Commented code
    --------------
    plt.figure(figsize=(8,8))
    img /=img.max()
    img = (img>0).astype(float)
    img = img.reshape((img.shape[0], img.shape[1]))
    shape = img.shape
    """
    plot_configs = get_plot_configs()
    plot_configs["xtick.labelsize"] = 8
    plot_configs["ytick.labelsize"] = 8
    plt.rcParams.update(plot_configs)
    fig = plt.figure(figsize=(9 * 3 / 5, 9), constrained_layout=True)
    gspec = fig.add_gridspec(5, 3, wspace=0.15, hspace=0.15)
    cmap = plt.cm.get_cmap("gnuplot", 5)
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
    for idx in range(5):
        for jdx in range(3):
            axis = fig.add_subplot(gspec[idx, jdx], autoscale_on=False)
            with h5py.File(files[ctr], "r") as fin:
                key_list = list(fin.keys())
                iter_n = get_ds_iters(key_list)
                img = fin[f"conf_{iter_n[-2]}"]
                axis.matshow(img, cmap=cmap)
            axis.set_xlim((0, 120))
            axis.set_ylim((0, 120))
            if jdx != 0:
                axis.set_yticklabels([])
            if idx != 0:
                axis.set_xticklabels([])
            ctr += 1
    fig.supylabel(r"Tumbling rate, $\alpha$")
    fig.supxlabel(r"Density, $\phi / \rm unit^{{-2}}$")
    fig.savefig("../plots/grid.pdf")
    fig.savefig("../plots/grid.png")


if __name__ == "__main__":
    main()
