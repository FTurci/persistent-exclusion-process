import lattice
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Nx = Ny = 32
Np = int(0.05*Nx*Ny)
tumble = 0.001
snapshot = int(1/tumble)
speed = 10
niter = int(100*snapshot)
print("Total number of iterations is",niter)
L = lattice.Lattice(Nx*Ny,Np)
L.set_square_connectivity(Nx,Ny)
L.reset_random_occupancy()
L.reset_orientations()


for iteration in range(niter):
    if iteration%snapshot==0
    L.c_move(tumble,speed)

