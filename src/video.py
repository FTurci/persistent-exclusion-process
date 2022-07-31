import lattice
import numpy as  np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Nx = Ny = 128
Np = int(0.05*Nx*Ny)
tumble = 0.1
speed = 10
L = lattice.Lattice(Nx*Ny,Np)
L.set_square_connectivity(Nx,Ny)
L.reset_random_occupancy()
L.reset_orientations()


# L.c_move(tumble)
# print(L.occupancy)

# print(L.neighbor_table[])
# print(L.neighbors(2))
# print(L.particles)

plt.rcParams["figure.autolayout"]=False
inch = 6
fig, ax = plt.subplots(figsize=(inch,inch),)
ax.set_facecolor('k')

scat = plt.scatter(*L.positions(),c=L.orientation,facecolors=L.orientation, s=(inch/Nx*21)**2, cmap=plt.cm.Set2, alpha=0.9)

ors =[]
def init():
    ax.set_xlim(0, Nx)
    ax.set_ylim(0,Ny)
    ax.set_aspect('equal', 'box')
    return scat,
def update(frame):
    L.c_move(tumble,speed)
    # scat.set_data(*L.positions())
    scat.set_offsets(np.array([*L.positions()]).T)
    scat.set_array(L.orientation)
    ors.append(np.average(L.orientation))
    print(frame,np.mean(ors))

    return scat,


# for l in range(3):
    # L.c_move(tumble)
ani = FuncAnimation(fig, update, frames=range(50),init_func=init,blit=True)
ani.save('myAnimation.gif', writer='imagemagick', fps=6)
plt.show()