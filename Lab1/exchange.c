/**********************************************************************
 * Point-to-point communication using MPI
 *
 **********************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int rank, size;
  double a, b;

  MPI_Init(&argc, &argv);               /* Initialize MPI               */
  MPI_Comm_size(MPI_COMM_WORLD, &size); /* Get the number of processors */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* Get my number                */
  
  a = 100.0 + (double) rank;  /* Different a on different processors */

  MPI_Request request_send, request_recv;
  MPI_Status status;

  /* Exchange variable a in a circular fashion using non-blocking communication */
  int tag = 111; // Use a single tag for both send and receive operations
  MPI_Isend(&a, 1, MPI_DOUBLE, (rank+1)%size,tag, MPI_COMM_WORLD, &request_send);
  MPI_Irecv(&b, 1, MPI_DOUBLE, (rank+size-1)%size, tag, MPI_COMM_WORLD, &request_recv);
  MPI_Wait(&request_send, &status); // Wait for the send operation to complete
  MPI_Wait(&request_recv, &status); // Wait for the receive operation to complete

  printf("Processor %d got %f from processor %d\n", rank, b, (rank+size-1)%size);

  MPI_Finalize(); 

  return 0;
}
