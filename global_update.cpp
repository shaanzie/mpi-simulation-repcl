#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#include "replay-clock.h"

#define N 10  // Number of iterations per process

class Packet {
    int seq_no;
    ReplayClock rc;
    int global_var;

    public:

        Packet(int seq_no, ReplayClock rc, int global_var) {
            seq_no = seq_no;
            rc = rc;
            global_var = global_var;
        }

        ReplayClock getReplayClock()
        {
            return rc;
        } 

        int getGlobalVar()
        {
            return global_var;
        }
};

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

    auto now = std::chrono::system_clock::now();
    ReplayClock rc = ReplayClock((uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count(), rank, EPSILON, INTERVAL);

    // Each process increments the global variable multiple times (N iterations)
    for (int i = 0; i < N; i++) {
        increment_global_variable(global_var, rank);


        rc.SendLocal((uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
        Packet p = Packet(i, rc, global_var);

        // Send updated value to the next process
        if (rank != size - 1) {
            MPI_Send(&p, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }

        // Receive updated value from the previous process
        if (rank != 0) {
            MPI_Recv(&p, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rc.Recv(p.getReplayClock(), (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count());
            global_var = p.getGlobalVar();
        }
    }

    // At the end of each process, print the value of the global variable
    std::cout << "Process " << rank << " has final global variable value: " << global_var << std::endl;

    MPI_Finalize();
    return 0;
}
