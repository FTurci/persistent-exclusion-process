#!/usr/bin/env python
"""Make a animated GIF of the lattice

Usage:
-----
./video.py

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import lattice


def main():
    """
    Initiate a lattice [class], evolve it, and save as a gif using FuncAnimation
    (which requires init() and update(), as defined within)
    """
    n_x = n_y = 128
    n_p = int(0.05 * n_x * n_y)
    tumble = 0.1
    speed = 10
    lat = lattice.Lattice(n_x * n_y, n_p)
    lat.set_square_connectivity(n_x, n_y)
    lat.reset_random_occupancy()
    lat.reset_orientations()
    plt.rcParams["figure.autolayout"] = False
    inch = 6
    fig, axis = plt.subplots(
        figsize=(inch, inch),
    )
    axis.set_facecolor("k")
    sct = plt.scatter(
        *lat.positions(),
        c=lat.orientation,
        facecolors=lat.orientation,
        s=(inch / n_x * 21) ** 2,
        cmap=plt.cm.Set2,  # TODO: Module has no Set2 member
        alpha=0.9
    )
    ori = []

    def init():
        axis.set_xlim(0, n_x)
        axis.set_ylim(0, n_y)
        axis.set_aspect("equal", "box")
        return (sct,)

    def update(frame):
        lat.c_move(tumble, speed)
        sct.set_offsets(np.array([*lat.positions()]).T)
        sct.set_array(lat.orientation)
        ori.append(np.average(lat.orientation))
        print(frame, np.mean(ori))
        return (sct,)

    ani = FuncAnimation(fig, update, frames=range(50), init_func=init, blit=True)
    ani.save("myAnimation.gif", writer="imagemagick", fps=6)
    plt.show()


if __name__ == "__main__":
    main()
