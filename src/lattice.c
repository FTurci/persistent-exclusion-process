// gcc -shared -W -o lattice.so -fPIC lattice.c
#include "stdio.h"

void _construct_square_neighbor_table(int* table ,int Nx,int Ny){
    
    const int connectivity = 4;
    unsigned int lattice_index;
    unsigned int u;


    for (int i=0;i<Nx;i++){
        for (int j=0;j<Ny; j++){
            
            lattice_index = i*Ny+j;
            u = lattice_index*connectivity;
            // up
            if (i==0){

                table[u+0] = (Nx-1)*Ny+j;
            }   
            else{
                table[u+0] = lattice_index-Nx;
            }
            // right
            if (j==Ny-1){
                table[u+1] = lattice_index-Ny+1;
         
            }   
            else{
                table[u+1] = lattice_index+1;
            }
             // down
            if (i==Nx-1){

                table[u+2] = j;
            }   
            else{
                table[u+2] = lattice_index+Nx;
            }
            // left
            if (j==0){

                table[u+3] = lattice_index+Ny-1;
            }   
            else{
                table[u+3] = lattice_index-1;

                // printf("Here %i\n", index);
            }

            lattice_index +=1;
        }

    }


}