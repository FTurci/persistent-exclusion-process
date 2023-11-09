"""
Utility functions
"""
import re
import h5py
import numpy as np


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


def get_mean_orientation(file) -> list:
    """
    Get the mean orientation at each iteration

    :param file: the h5 file to open [str]
    :returns: mean orientation [list] of length 1000

    Go through all iteration
    """
    hf = h5py.File(file, "r")
    key_list = list(hf.keys())
    iter_n = get_ds_iters(key_list)
    ori = []
    ori_acm = []
    for _, val in enumerate(iter_n):
        sshot = np.array(hf[f"conf_{val}"]).flatten()
        avg_ori = np.average(sshot[np.where(sshot != 0)[0]] - 1)
        ori.append(avg_ori)
        ori_acm.append(np.mean(ori))
    return ori_acm
