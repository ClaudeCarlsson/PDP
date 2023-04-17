#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//mpirun -n 16 ./matmul input.txt output.txt

int main(int argc, char *argv[]) 
{
    // Initialize variables for MPI
    int rank, size, n;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the correct number of arguments is provided (For process 0)
    if (argc != 3)
    {
        if (rank == 0)
        {
            printf("Usage: %s <input_file> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Initialize variables for the matrix multiplication
    double *A = NULL, *B = NULL, *C = NULL;

    // Read the input file and allocate memory for matrices A and B (For process 0)
    if (rank == 0)
    {
        // Open file
        FILE *input = fopen(argv[1], "r");

        // Read n from input file
        int n;
        if (fscanf(input, "%d", &n) != 1)
        {
            printf("Error when reading n from input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate and read matrix A from input file
        A = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++)
        {
            if (fscanf(input, "%lf", &A[i]) != 1)
            {
                printf("Error when reading A from input file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Allocate and read matrix B from input file
        B = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++)
        {
            if (fscanf(input, "%lf", &B[i]) != 1)
            {
                printf("Error when reading B from input file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Close file
        fclose(input);
    }

    // Start the timer
    double start = MPI_Wtime();

    // Broadcast the matrix size to all processes from process 0
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate the memory for matrices A and B (For process not 0)
    if (rank != 0)
    {
        A = (double *)malloc(n * n * sizeof(double));
        B = (double *)malloc(n * n * sizeof(double));
    }

    // Allocate memory for the result matrix C
    C = (double *)calloc(n * n, sizeof(double));

    // Broadcast the matrices A and B to all processes
    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // The matrix multiplication, partitioning done row-wise
    for (int i = rank; i < n; i += size) 
    // i: current row of matrix A 
    {
        for (int j = 0; j < n; j++) 
        // j: current column of matrix B
        {
            for (int k = 0; k < n; k++) 
            // k: current index being multiplied and summed for the row of matrix A and column of matrix B
            {
                // C[i][j] += A[i][k] and B[k][j] 
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    // Reduce the partial results to get the final result (For process 0)
    if (rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, C, n * n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(C, C, n * n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Stop the timer
    double end = MPI_Wtime();

    // Write the resulting matrix C to the output file and print the execution time (For process 0)
    if (rank == 0)
    {
        // Open the file 
        FILE *output = fopen(argv[2], "w");

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fprintf(output, "%.6lf ", C[i * n + j]);
            }
            fprintf(output, "\n");
        }

        // Close the file
        fclose(output);

        // Print the time
        printf("%.6lf\n", end - start);
    }

    // Free allocated memory for matrices A, B, and C
    free(A);
    free(B);
    free(C);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}
