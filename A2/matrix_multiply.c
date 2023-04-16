#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    // Initialize variables
    int rank, size;
    int n;
    double *A= NULL, *B= NULL, *C = NULL;

    // Initialize the MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check if the correct number of arguments is provided
    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s input_file output_file\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Read input file and store matrices A and B (only on the root process)
    if (rank == 0) {
        FILE *input = fopen(argv[1], "r");
        if (fscanf(input, "%d", &n) != 1) {
            perror("Error reading n from input file");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        A = (double *)malloc(n * n * sizeof(double));
        B = (double *)malloc(n * n * sizeof(double));
        for (int i = 0; i < n * n; i++) {
            if (fscanf(input, "%lf", &A[i]) != 1) {
                perror("Error reading A from input file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        for (int i = 0; i < n * n; i++) {
            if (fscanf(input, "%lf", &B[i]) != 1) {
                perror("Error reading B from input file");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        fclose(input);
    }

    // Broadcast the matrix size to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for matrices A and B on all non-root processes
    if (rank != 0) {
        A = (double *)malloc(n * n * sizeof(double));
        B = (double *)malloc(n * n * sizeof(double));
    }

    // Broadcast the matrices A and B to all processes
    MPI_Bcast(A, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for the result matrix C
    C = (double *)calloc(n * n, sizeof(double));

    // Start the timer
    double start = MPI_Wtime();

    // Perform matrix multiplication in parallel
    for (int i = rank; i < n; i += size) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }

    // Reduce the partial results to obtain the final result 
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, C, n * n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        MPI_Reduce(C, C, n * n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    // Stop the timer
    double end = MPI_Wtime();

    // Write the resulting matrix C to the output file and print the 
    // execution time (only on the root process)
    if (rank == 0) {
        FILE *output = fopen(argv[2], "w");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(output, "%.6lf ", C[i * n + j]);
            }
            fprintf(output, "\n");
        }
        fclose(output);

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
