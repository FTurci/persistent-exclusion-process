import numpy as np
import h5py
import glob
from stringato import extract_floats
import matplotlib.pyplot as plt



def data_load():
    files = glob.glob("../data/dataset*")

    inputs,outputs = [],[]
    for f in files:
        tumble = float(extract_floats(f)[0])
        
        with h5py.File(f, "r") as fin:
            count=0
            for key in fin.keys():
                img = fin[key][:]
                # img /=img.max()
                # img = (img>0).astype(float)
                # img = img.reshape((img.shape[0], img.shape[1]))
                shape = img.shape
                inputs.append(img)
                outputs.append(tumble)
                # plt.figure(figsize=(8,8))
                plt.matshow(img)
                plt.title(str(tumble))
                plt.colorbar()
                plt.show()
                count+=1
                if count>0:
                    break
data_load()