#include "stencil.h"


int main(int argc, char **argv) {
	// MPI setup
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (4 != argc) {
		printf("Usage: stencil input_file output_file number_of_applications\n");
		return 1;
	}
	char *input_name = argv[1];
	char *output_name = argv[2];
	int num_steps = atoi(argv[3]);

	// Read input file
	double *input;
	int num_values;
	if (0 > (num_values = read_input(input_name, &input))) {
		return 2;
	}

	// Stencil values
	double h = 2.0*PI/num_values;
	const int STENCIL_WIDTH = 5;
	const int EXTENT = STENCIL_WIDTH/2;
	const double STENCIL[] = {1.0/(12*h), -8.0/(12*h), 0.0, 8.0/(12*h), -1.0/(12*h)};

	// Calculate local work size
	int local_num_values = num_values / size;
	int local_start = rank * local_num_values;
	int local_end = local_start + local_num_values;

	if (rank == size - 1) {
		local_end = num_values;
	}
	local_num_values = local_end - local_start;

	// Start timer
	double start = MPI_Wtime();

	// Allocate data for local result
	double *local_output;
	if (NULL == (local_output = malloc(local_num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for local output");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}


	// Repeatedly apply stencil
	for (int s = 0; s < num_steps; s++) {
		// Apply stencil
		for (int i = local_start; i < local_start + EXTENT; i++) {
			double result = 0;
			for (int j = 0; j < STENCIL_WIDTH; j++) {
				int index = (i - EXTENT + j + num_values) % num_values;
				result += STENCIL[j] * input[index];
			}
			local_output[i - local_start] = result;
		}
		for (int i = local_start + EXTENT; i < local_end - EXTENT; i++) {
			double result = 0;
			for (int j = 0; j < STENCIL_WIDTH; j++) {
				int index = i - EXTENT + j;
				result += STENCIL[j] * input[index];
			}
			local_output[i - local_start] = result;
		}
		for (int i = local_end - EXTENT; i < local_end; i++) {
			double result = 0;
			for (int j = 0; j < STENCIL_WIDTH; j++) {
				int index = (i - EXTENT + j) % num_values;
				result += STENCIL[j] * input[index];
			}
			local_output[i - local_start] = result;
		}
		// Swap input and output
		if (s < num_steps - 1) {
			double *tmp = input;
			input = local_output;
			local_output = tmp;
		}
	}


	// Gather results
	double *output = NULL;
	if (rank == 0) {
		output = malloc(num_values * sizeof(double));
	}

	MPI_Gather(local_output, local_num_values, MPI_DOUBLE, output, local_num_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	// Stop timer
	double my_execution_time = MPI_Wtime() - start;
	double max_execution_time;
	MPI_Reduce(&my_execution_time, &max_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


	// Write result
	if (rank == 0) {
		printf("Time: %f \n", max_execution_time);
#ifdef PRODUCE_OUTPUT_FILE
		if (0 != write_output(output_name, output, num_values)) {
			MPI_Abort(MPI_COMM_WORLD, 2);
		}
#endif
	}

	// Clean up
	free(output);

	// Finalize MPI
	MPI_Finalize();

	return 0;
}


int read_input(const char *file_name, double **values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "r"))) {
		perror("Couldn't open input file");
		return -1;
	}
	int num_values;
	if (EOF == fscanf(file, "%d", &num_values)) {
		perror("Couldn't read element count from input file");
		return -1;
	}
	if (NULL == (*values = malloc(num_values * sizeof(double)))) {
		perror("Couldn't allocate memory for input");
		return -1;
	}
	for (int i=0; i<num_values; i++) {
		if (EOF == fscanf(file, "%lf", &((*values)[i]))) {
			perror("Couldn't read elements from input file");
			return -1;
		}
	}
	if (0 != fclose(file)){
		perror("Warning: couldn't close input file");
	}
	return num_values;
}


int write_output(char *file_name, const double *output, int num_values) {
	FILE *file;
	if (NULL == (file = fopen(file_name, "w"))) {
		perror("Couldn't open output file");
		return -1;
	}
	for (int i = 0; i < num_values; i++) {
		if (0 > fprintf(file, "%.4f ", output[i])) {
			perror("Couldn't write to output file");
		}
	}
	if (0 > fprintf(file, "\n")) {
		perror("Couldn't write to output file");
	}
	if (0 != fclose(file)) {
		perror("Warning: couldn't close output file");
	}
	return 0;
}
