"""Lattice class"""

import ctypes #a library that provides C data types and allows calling functions in DLLs or shared libraries

import numpy as np

_clattice = ctypes.CDLL("c/lattice.so") #Compile Dynamic Link Libraries; we use this to load lattice.so (so = shared object; a DLL is a shared object that doesn't need to be there at compilation time; this is not relevant to interpreted languages like Python ?)
c_int_p = ctypes.POINTER(ctypes.c_int) #set c_int_p as an int pointer


class Lattice: #Lattice defines the class, of which particular instances are 'objects'
    def __init__(self, Nsites, Nparticles, connectivity=4): #initialises object's attributes when initialising/defining an object
        self.Nsites = Nsites #Nsites is the number of sites on lattice
        self.Nparticles = Nparticles #Nparticles is the number of particles on lattice
        self.connectivity = connectivity # 

    def reset_random_occupancy(self):
        self.occupancy = np.zeros(self.Nsites, dtype=np.int32)
        self.occupancy[
            np.random.choice(self.Nsites, self.Nparticles, replace=False)
        ] = 1
        # the order matters
        self.particles = np.where(self.occupancy > 0)[0].astype(dtype=np.int32)

    def reset_orientations(self):
        # same order as particles
        self.orientation = np.random.randint(
            0, self.connectivity, size=self.Nparticles, dtype=np.int32
        )

    def set_square_connectivity(self, Nx, Ny):
	'''
	Program forces Nx,Ny dimensions to match number of lattice sites. Creates (and then flattens) a 2D array listing empty rows of Nsites and empty columns of connectivities; this is the list of neighbours that each site has. They are then filled by construct_square_neighbor_table. Presumably, the neighbour table links each specific lattice site with its neighbours for later examination.

	Nx = number of sites in lattice along x direction
	Ny = number of sites in lattice along y direction
	'''
        assert Nx * Ny == self.Nsites, "Nx and Ny are incorrect."
        self.Nx = Nx
        self.Ny = Ny
        neighbor_table = np.zeros(
            (self.Nsites, self.connectivity), dtype=np.int32
        ).flatten()

        _clattice._construct_square_neighbor_table(
            neighbor_table.ctypes.data_as(c_int_p), Nx, Ny
        )
	#self.neighbor_table_flat = neighbor_table (why not do this instead of reflattening afterwards?
        self.neighbor_table = neighbor_table.reshape(self.Nsites, self.connectivity)
        self.neighbor_table_flat = neighbor_table.flatten() #and delete this?

    def neighbors(self, site):
	'''
	Defines neighbour table attribute.
	'''
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
        if self.occupancy[attempt] == 0:
            self.occupancy[site] = 0
            self.occupancy[attempt] = 1
            self.particles[pick] = attempt

        if np.random.uniform(0, 1) < tumble_probability:
            self.orientation[pick] = np.random.randint(0, self.connectivity)

    def positions(self):
        x, y = np.unravel_index(self.particles, shape=(self.Nx, self.Ny))
        return x, y

    def c_move(self, tumble_probability, speed):
        _clattice._move(
            4,
            self.Nparticles,
            self.neighbor_table_flat.ctypes.data_as(c_int_p),
            self.orientation.ctypes.data_as(c_int_p),
            self.occupancy.ctypes.data_as(c_int_p),
            self.particles.ctypes.data_as(c_int_p),
            ctypes.c_double(tumble_probability),
            speed,
        )

        assert len(self.particles) == self.Nparticles, "not ok"

    def image(self):
        matrix = np.zeros((self.Nx, self.Ny))
        x, y = self.positions()
        matrix[x, y] = self.orientation
        return matrix
