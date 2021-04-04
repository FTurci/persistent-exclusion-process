import lattice
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
import tqdm

Nx = Ny = 32
Np = int(0.05*Nx*Ny)
for tumble in tqdm.tqdm(np.arange(0.001,1, 0.01)):
    print("Tumble", tumble)
    snapshot = int(1/tumble)
    speed = 10
    niter = int(300*snapshot)
    print("Total number of iterations is",niter)
    L = lattice.Lattice(Nx*Ny,Np)
    L.set_square_connectivity(Nx,Ny)
    L.reset_random_occupancy()
    L.reset_orientations()
    # warmup
    for w in range(500):
        L.c_move(tumble,speed)

    with h5py.File(f"../data/dataset_tumble_{np.log10(tumble):.4f}.h5","w") as fout:
        for iteration in range(niter):
            L.c_move(tumble,speed)
            if (iteration%snapshot)==0:
                fout.create_dataset(f'conf_{iteration}', data=L.image().astype(np.int32))
        

