# Persistent Exclusion Process

This is a `Python` and `C` implementation of the persistent exclusion process
(see [Sepulveda and Soto PRL
2017](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.078001)),
using `ctypes`.

---

## Installation

On cluster

```bash
# Clone onto machine
git clone https://github.com/dlactivematter/persistent-exclusion-process
cd persistent-exclusion-process

# Prepare local environment
mkdir data
sh bp_env.sh

# Compile the lattice ctype
cd src/c
make
```

In the folder `src` one can find all the Python scripts.
