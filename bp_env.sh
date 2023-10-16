#!/bin/bash

for pkg in h5py tqdm matplotlib numpy keras protobuf==3.6.1 tensorflow
do
    ins=$(pip3 list --format legacy | grep -c "$pkg")
    if [ "$ins" -eq 1 ]
    then
        continue
    fi
	pip3 install --user "$pkg"
done

module load lang/gcc/12.3.0
