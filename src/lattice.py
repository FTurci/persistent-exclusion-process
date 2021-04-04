import ctypes
import numpy as np


_clattice = ctypes.CDLL("c/lattice.so")
c_int_p = ctypes.POINTER(ctypes.c_int)


class Lattice:
    def __init__(self, Nsites, Nparticles, connectivity=4):
        self.Nsites = Nsites
        self.Nparticles = Nparticles
        self.connectivity = connectivity

    def reset_random_occupancy(self):
        self.occupancy = np.zeros(self.Nsites, dtype=np.int32)
        self.occupancy[np.random.choice(self.Nsites, self.Nparticles, replace=False)]=1
        # the order matters
        self.particles = np.where(self.occupancy>0)[0].astype(dtype=np.int32)

    def reset_orientations(self):
        # same order as particles
        self.orientation = np.random.randint(0,self.connectivity,size=self.Nparticles,dtype=np.int32)

    def set_square_connectivity(self, Nx, Ny):

        assert Nx*Ny == self.Nsites, "Nx and Ny are incorrect."
        self.Nx = Nx
        self.Ny = Ny
        neighbor_table = np.zeros((self.Nsites, self.connectivity),dtype=np.int32).flatten()

        _clattice._construct_square_neighbor_table(
            neighbor_table.ctypes.data_as(c_int_p), 
            Nx,
            Ny)

        self.neighbor_table = neighbor_table.reshape(self.Nsites,self.connectivity)
        self.neighbor_table_flat = neighbor_table.flatten();

    def neighbors(self, site):
        return self.neighbor_table[site]

    def move(self, tumble_probability):
        # pick particle
        pick = np.random.randint(self.Nparticles)
        site = self.particles[pick]
        # pick neighbour following the orientation
        # attempt = np.random.choice(self.neighbors(site))
        o = self.orientation[pick]
        # print("orient",o)
        attempt = self.neighbors(site)[o]
        # print(pick,"attempt",attempt, self.occupancy[attempt])
        if self.occupancy[attempt]==0:
            self.occupancy[site] = 0
            self.occupancy[attempt] = 1
            self.particles[pick] = attempt

        if np.random.uniform(0,1)<tumble_probability:
            self.orientation[pick] = np.random.randint(0,self.connectivity)


    def positions(self):
        x,y = np.unravel_index(self.particles,shape = (self.Nx, self.Ny))
        return x,y

    def c_move(self,tumble_probability,speed):
        _clattice._move(4,
            self.Nparticles,
            self.neighbor_table_flat.ctypes.data_as(c_int_p),
            self.orientation.ctypes.data_as(c_int_p),
            self.occupancy.ctypes.data_as(c_int_p),
            self.particles.ctypes.data_as(c_int_p),
            ctypes.c_double(tumble_probability),
            speed
            )

        assert len(self.particles)==self.Nparticles, "not ok"
