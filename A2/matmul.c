#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

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

        // Transpose matrix B
        double *transposeB = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                transposeB[j * n + i] = B[i * n + j];
            }
        }
        free(B);
        B = transposeB;
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

    // Distribute the matrices A and B to all processes using MPI_Scatterv
    int rows_per_process = n / size;

    int *sendcounts_A = (int *)malloc(size * sizeof(int));
    int *displs_A = (int *)malloc(size * sizeof(int));
    int *sendcounts_B = (int *)malloc(size * sizeof(int));
    int *displs_B = (int *)malloc(size * sizeof(int));
    int *recvcounts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));

    int rows_sent = 0;
    for (int i = 0; i < size; i++)
    {
        int current_rows_per_process = n / size + (i < (n % size) ? 1 : 0);
        sendcounts_A[i] = current_rows_per_process * n;
        displs_A[i] = rows_sent * n;
        sendcounts_B[i] = current_rows_per_process * n;
        displs_B[i] = rows_sent * n;
        recvcounts[i] = current_rows_per_process * n;
        displs[i] = rows_sent * n;
        rows_sent += current_rows_per_process;
    }



    double *local_A = (double *)malloc(rows_per_process * n * sizeof(double));
    double *local_B = (double *)malloc(rows_per_process * n * sizeof(double));

    MPI_Scatterv(A, sendcounts_A, displs_A, MPI_DOUBLE, local_A, n * n / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(B, sendcounts_B, displs_B, MPI_DOUBLE, local_B, n * n / size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print the parts of the matrices each rank has
    for (int r = 0; r < size; r++)
    {
        if (rank == r)
        {
            printf("Rank %d has the following part of matrix A:\n", rank);
            for (int i = 0; i < rows_per_process; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    printf("%.2lf ", local_A[i * n + j]);
                }
                printf("\n");
            }
            printf("\n");

            printf("Rank %d has the following part of matrix B:\n", rank);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < rows_per_process; j++)
                {
                    printf("%.2lf ", local_B[i * rows_per_process + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
        // Synchronize the processes to ensure proper print order
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0) 
    {
        printf("rows_per_process: %d\n", rows_per_process);
    }
        

    // The matrix multiplication
    for (int r = 0; r < size; r++)
    {
        if (rank == r)
        {
            for (int i = 0; i < rows_per_process; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        printf("Rank %d: local_A[%d][%d] = %.2lf * local_B[%d][%d] = %.2lf\n", rank, i, k, local_A[i * n + k], j, k, local_B[j * n + k]);
                        C[i * n + j] += local_A[i * n + k] * local_B[j * n + k];
                    }
                    printf("Rank %d: Current index for C: C[%d][%d] = C[%d]\n", rank, i, j, i * n + j);
                }
            }
        }
        // Synchronize the processes to ensure proper print order
        MPI_Barrier(MPI_COMM_WORLD);
    }


    // Gather the results using MPI_Gatherv
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(C, n * rows_per_process, MPI_DOUBLE, C, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

    // Free allocated memory for sendcounts and displacements
    free(sendcounts_A);
    free(displs_A);
    free(sendcounts_B);
    free(displs_B);
    free(recvcounts);
    free(displs);

    // Free allocated memory for local matrices
    free(local_A);
    free(local_B);

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}