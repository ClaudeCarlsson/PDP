#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char *argv[]) {
  int rank, size, i, target_rank;
  double a;
  MPI_Status status;

  MPI_Init(&argc, &argv);               /* Initialize MPI               */
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* Get the number of processors */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get my number                */

  /* Ensure the number of processes is 2^k */
  if (size & (size - 1)) {
    if (rank == 0) {
      printf("Error: Number of processes must be a power of 2.\n");
    }
    MPI_Finalize();
    return 1;
  }

  /* Fan-out sequence */
  if (rank == 0) {
    a = 999.999;
  }

  int k = log2(size);
  for (i = 0; i < k; ++i) {
    if (rank % (1 << (i + 1)) == 0) {
      target_rank = rank + (1 << i);
      if (i > 0) {
        MPI_Recv(&a, 1, MPI_DOUBLE, rank - (1 << i), 111, MPI_COMM_WORLD, &status);
      }
      if (target_rank < size) {
        MPI_Send(&a, 1, MPI_DOUBLE, target_rank, 111, MPI_COMM_WORLD);
      }
    }
  }

  if (rank > 0) {
    MPI_Recv(&a, 1, MPI_DOUBLE, rank - (1 << (i - 1)), 111, MPI_COMM_WORLD, &status);
  }

  printf("Processor %d got %f\n", rank, a);

  MPI_Finalize();

  return 0;
}
