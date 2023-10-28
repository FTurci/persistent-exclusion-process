#!/bin/bash

module load lang/gcc/12.3.0
module load lang/python/3.9.13

for pkg in h5py tqdm matplotlib numpy keras pandas tensorflow
do
    ins=$(pip3 list --format legacy | grep -c "$pkg")
    if [ "$ins" -ne 1 ]
    then
	  pip3 install --user "$pkg"
    fi
done
