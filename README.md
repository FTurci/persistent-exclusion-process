# Persistent Exclusion Process

This is a `Python` and `C` implementation of the persistent exclusion process
(see [Sepulveda and Soto PRL
2017](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.078001)),
using `ctypes`.

---

## Installation

On cluster (BLUEPEBLLE)

```bash
# Clone onto machine
git clone https://github.com/dlactivematter/persistent-exclusion-process
cd persistent-exclusion-process

# Prepare local environment
mkdir data
source .env_bp

# Compile the lattice ctype
cd src/c
make
```

In the folder `src` one can find all the Python scripts.

## Generating dataset

```bash
python3 sampler.py --density <DENSITY>
```

Add the flag `--odd` to produce the odd set of tumbling rate. Run both with and
without the flag to generate the full dataset.
