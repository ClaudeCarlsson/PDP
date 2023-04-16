/**********************************************************************
 * A simple "hello world" program for MPI/C
 *
 **********************************************************************/
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of this process

   if (rank == 0)
   {
      MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes
      printf("Total processes: %d\n", size);
   }

    printf("Hello from process: %d \n", rank);

    MPI_Finalize(); // Clean up MPI environment
    return 0;
}

