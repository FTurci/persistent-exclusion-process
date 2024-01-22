from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils
import cmcrameri #for different cmaps

relmaxclustersizes1 = [] #relative max cluster sizes
relmaxclustersizes2 = []

Pt=0.157 #tumble probability
Pt_array = [0.016,0.157]
rho_array = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

i=0
for Pt in Pt_array:
    for rho in rho_array: #particle density
        file = ("../data/dataset_tumble_{:.3f}_{}.h5".format(Pt,rho)) #change this to analyse different file
        hfile = h5py.File(file,"r")

        iters = utils.get_ds_iters(hfile.keys()) #get all iterations of evolution
        image = hfile[f"conf_{iters[-1]}"] #establish image as last iteration in evolution

        #cluster analysis
        kernel = [[0,1,0],
                [1,1,1],
                [0,1,0]]
        labelled, nlabels = ndimage.label(image,structure=kernel) #clusters are now labelled in 2D form

        cluster_sizes = np.bincount(labelled.flatten())[1:] #sizes of clusters now stored in 1D form
        biggest_cluster = np.max(cluster_sizes)

        map_size = 128*128
        ratio = biggest_cluster/map_size
        percentage = ratio*100
        if i == 0:
            relmaxclustersizes1.append(percentage)
        else:
            relmaxclustersizes2.append(percentage)
    i+=1

fig, (ax1,ax2) = plt.subplots(2,1,figsize = (10,6))
ax1.scatter(rho_array,relmaxclustersizes1)
ax1.set_xlabel(r"System Density $\rho$")
ax1.set_ylabel("Biggest Cluster Percentage")
ax1.set_title(r"Biggest Cluster Sizes Relative to System Size Against System Density for Tumbling Rate 0.016")
ax2.scatter(rho_array,relmaxclustersizes2)
ax2.set_xlabel(r"System Density $\rho$")
ax2.set_ylabel ("Biggest Cluster Percentage")
ax2.set_title("Biggest Cluster Sizes Relative to System Size Against System Density for Tumbling Rate 0.157")
plt.show()


