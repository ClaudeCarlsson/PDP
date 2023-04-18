#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <mpi.h>
#include<unistd.h>

void print_block(double *block, int block_size, int rank) {
    printf("Rank %d:\n", rank);
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            printf("%.2lf ", block[i * block_size + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(int argc, char *argv[])
{
    // Declare MPI variables and initialize MPI
    int rank, size, n;
    MPI_Comm grid_comm, row_comm, col_comm;
    int grid_coords[2], row_rank, row_size, col_rank, col_size;
    int source, dest;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the correct number of arguments was provided
    if (argc != 3)
    {
        if (rank == 0)
        {
            printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int grid_dim = (int)sqrt(size);

    if (size != grid_dim * grid_dim)
    {
        if (rank == 0)
        {
            printf("Error: The number of processes must be a perfect square.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Declare pointers for the matrices and blocks
    double *A = NULL, *B = NULL, *C = NULL, *A_block = NULL, *B_block = NULL, *C_block = NULL;

    // If the current process is rank 0, read the input matrices from a file
    if (rank == 0)
    {
        FILE *input = fopen(argv[1], "r");

        // Read the size of the matrices from the input file
        if (fscanf(input, "%d", &n) != 1)
        {
            printf("Error when reading n from input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory for matrix A and read its values from the input file
        A = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++)
        {
            if (fscanf(input, "%lf", &A[i]) != 1)
            {
                printf("Error when reading A from input file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Allocate memory for matrix B and read its values from the input file
        B = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++)
        {
            if (fscanf(input, "%lf", &B[i]) != 1)
            {
                printf("Error when reading B from input file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Close the input file
        fclose(input);
    }


    // Record the start time
    double start = MPI_Wtime();

    // Broadcast the value of n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine the dimensions of the grid of processes and create the grid communicator
    int dims[2] = {grid_dim, grid_dim};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);

    // Determine the row and column coordinates of the current process within the grid
    MPI_Cart_coords(grid_comm, rank, 2, grid_coords);

    // Create the row and column communicators for the current process
    int row_remain_dims[2] = {0, 1};
    MPI_Cart_sub(grid_comm, row_remain_dims, &row_comm);
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);

    int col_remain_dims[2] = {1, 0};
    MPI_Cart_sub(grid_comm, col_remain_dims, &col_comm);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);

    // Compute the size of each block and allocate memory for the blocks
    int block_size = n / grid_dim;
    A_block = (double *)malloc(block_size * block_size * sizeof(double));
    B_block = (double *)malloc(block_size * block_size * sizeof(double));
    C_block = (double *)calloc(block_size * block_size, sizeof(double));

    
    MPI_Barrier(MPI_COMM_WORLD);

    // Check if the rank is 0
    if (rank == 0) {
        // Loop through the 2D grid of processes
        for (int i = 0; i < grid_dim; i++) {
            for (int j = 0; j < grid_dim; j++) {
                // Get the rank of the process at coordinates (i, j)
                int proc_rank;
                int coords[2] = {i, j};
                MPI_Cart_rank(grid_comm, coords, &proc_rank);

                // Copy the block of matrix corresponding to the (i, j) coordinate to a separate memory location
                for (int k = 0; k < block_size; k++) {
                    for (int l = 0; l < block_size; l++) {
                        // Compute the index of the current element in the original matrix A
                        int A_index = (i * block_size + k) * n + (j * block_size + l);
                        int B_index = (j * block_size + k) * n + (i * block_size + l);

                        // Copy the element from the original matrix to the corresponding location 
                        A_block[k * block_size + l] = A[A_index];
                        B_block[k * block_size + l] = B[B_index];
                    }
                }

                // Send the blocks of A and B to the process using MPI_Send
                MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, proc_rank, 0, MPI_COMM_WORLD);
                MPI_Send(B_block, block_size * block_size, MPI_DOUBLE, proc_rank, 1, MPI_COMM_WORLD);
            }
        }
    }
    // If the rank is not 0, receive the blocks of A and B from process 0

    // Receive the blocks of A and B from process 0 using MPI_Recv
    MPI_Recv(A_block, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(B_block, block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Print the blocks in each rank
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        print_block(A_block, block_size, rank);
        for (int i = 1; i < size; i++) {
            MPI_Recv(A_block, block_size * block_size, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_block(A_block, block_size, i);
        }
    } else {
        MPI_Send(A_block, block_size * block_size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        print_block(B_block, block_size, rank);
        for (int i = 1; i < size; i++) {
            MPI_Recv(B_block, block_size * block_size, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print_block(B_block, block_size, i);
        }
    } else {
        MPI_Send(B_block, block_size * block_size, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Loop through each stage of the algorithm
    for (int stage = 0; stage < grid_dim; stage++)
    {
        // Determine the root process for the broadcast operation in this stage
        int root = (grid_coords[0] + stage) % grid_dim;

        // Broadcast the current block of matrix A to all processes in the same row
        MPI_Bcast(A_block, block_size * block_size, MPI_DOUBLE, root, row_comm);

        // Perform the matrix multiplication for the current block
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                for (int k = 0; k < block_size; k++)
                {
                    C_block[i * block_size + j] += A_block[i * block_size + k] * B_block[k * block_size + j];
                }
            }
        }

        // Shift the block of matrix B to the left by one process in the current column
        MPI_Cart_shift(col_comm, 0, 1, &source, &dest);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE, dest, 0, source, 0, col_comm, MPI_STATUS_IGNORE);
    }


    // Check if the current process is rank 0
    if (rank == 0)
    {
        // Allocate memory for the full matrix C
        C = (double *)calloc(n * n, sizeof(double));

        // Copy the result from the top-left block to the corresponding part of C
        for (int i = 0; i < block_size; i++)
        {
            for (int j = 0; j < block_size; j++)
            {
                C[i * n + j] = C_block[i * block_size + j];
            }
        }

        // Receive the result blocks from each process and copy to C
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(C_block, block_size * block_size, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Get the row and column coordinates of the process
            int coords[2];
            MPI_Cart_coords(grid_comm, i, 2, coords);

            // Copy the block to the corresponding part of C
            for (int j = 0; j < block_size; j++)
            {
                for (int k = 0; k < block_size; k++)
                {
                    C[(coords[0] * block_size + j) * n + (coords[1] * block_size + k)] = C_block[j * block_size + k];
                }
            }
        }

        // Record the end time and write the result to a file
        double end = MPI_Wtime();

        FILE *output = fopen(argv[2], "w");

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fprintf(output, "%.6lf ", C[i * n + j]);
            }
            fprintf(output, "\n");
        }

        fclose(output);

        // Print the elapsed time
        printf("%.6lf\n", end - start);
    }
    // For all other processes, send the result block to process 0
    else
    {
        MPI_Send(C_block, block_size * block_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    }

    free(A);
    free(B);
    free(C);

    free(A_block);
    free(B_block);
    free(C_block);

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&grid_comm);

    MPI_Finalize();

    return 0;
}