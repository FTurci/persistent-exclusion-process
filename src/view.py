#!/usr/bin/env python
"""
Display lattice using pyplot

Usage:
-----
./view.py

"""

import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

# TODO: what is this??
from stringato import extract_floats


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
    cmap = plt.cm.get_cmap("gnuplot", 5)
    files = glob.glob("../data/dataset*")
    inputs, outputs = [], []
    for file in files:
        tumble = float(extract_floats(file)[0])
        with h5py.File(file, "r") as fin:
            count = 0
            for _, val in fin.items():
                img = val
                inputs.append(img)
                outputs.append(tumble)
                plt.matshow(img, cmap=cmap)
                plt.title(str(tumble))
                plt.colorbar()
                plt.show()
                count += 1
                print(np.unique(img))
                if count > 0:
                    break


if __name__ == "__main__":
    main()
