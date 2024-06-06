from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils
import cmcrameri #for different cmaps

relmaxclustersizes1 = [] #relative max cluster sizes
relmaxclustersizes2 = []
relmaxclustersizes3 = []
relmaxclustersizes4 = []
relmaxclustersizes5 = []

Pt_array = [0.016,0.034,0.073,0.157,0.340] #tumble probability values
rho_array = [0.025,0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95,0.975] #density values

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
        if i == 0:
            relmaxclustersizes1.append(ratio)
        elif i == 1:
            relmaxclustersizes2.append(ratio)
        elif i == 2:
            relmaxclustersizes3.append(ratio)
        elif i == 3:
            relmaxclustersizes4.append(ratio)
        elif i == 4:
            relmaxclustersizes5.append(ratio)
    i+=1

fig, ax = plt.subplots(5,1,figsize = (10,16))
ax.flatten()
ax[0].scatter(rho_array,relmaxclustersizes1)
ax[1].scatter(rho_array,relmaxclustersizes2)
ax[2].scatter(rho_array,relmaxclustersizes3)
ax[3].scatter(rho_array,relmaxclustersizes4)
ax[4].scatter(rho_array,relmaxclustersizes5)

for i in [0,1,2,3,4]:
    ax[i].set_xlabel(r"$\rho$")
    ax[i].set_ylabel("Ratio")
    ax[i].set_title("Tumbling Rate {}".format(Pt_array[i]))
plt.suptitle("Biggest Cluster Sizes Relative to System Size Against System Density for Different Tumbling Rates")
fig.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig("biggest-cluster-sizes.png")
plt.show()


