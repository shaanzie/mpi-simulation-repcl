#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define N 10  // Number of iterations per process

// Function to increment the global variable
void increment_global_variable(int &global_var, int rank) {
    int local_incr = rank + 1;  // Each process tries to add its rank + 1 to the global variable
    global_var += local_incr;
    std::cout << "Process " << rank << " incremented global var by " << local_incr << std::endl;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int global_var = 0;  // Shared global variable to be incremented by all processes

    // Each process increments the global variable multiple times (N iterations)
    for (int i = 0; i < N; i++) {
        increment_global_variable(global_var, rank);

        // Send updated value to the next process
        if (rank != size - 1) {
            MPI_Send(&global_var, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }

        // Receive updated value from the previous process
        if (rank != 0) {
            MPI_Recv(&global_var, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // At the end of each process, print the value of the global variable
    std::cout << "Process " << rank << " has final global variable value: " << global_var << std::endl;

    MPI_Finalize();
    return 0;
}
