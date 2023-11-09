CC=gcc
CFLAGS=-shared -W -O3 -fPIC
SRC=lattice.c pcg_basic.c
OBJ=lattice.so

lattice.so: $(SRC)
	$(CC) $(CFLAGS) -o $(OBJ) $(SRC) 
