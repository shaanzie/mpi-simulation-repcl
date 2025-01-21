#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>

#include "replay-clock.h"

using namespace std;

#define N 10  // Number of iterations per process

class Packet {

    public:

        int seq_no;
        ReplayClock rc;
        int global_var;

        Packet(uint32_t s, ReplayClock c, uint32_t g) {
            seq_no = s;
            rc = c;
            global_var = g;
        }

        ReplayClock getReplayClock()
        {
            return rc;
        } 

        uint32_t getGlobalVar()
        {
            return global_var;
        }
};

std::vector<char> serialize(Packet &p)
{
    std::vector<char> buffer(sizeof(uint32_t) * 6);
    char* ptr = buffer.data();
    
    uint32_t hlc = p.rc.GetHLC();
    std::memcpy(ptr, &hlc, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    uint32_t offsetbitmap = p.rc.GetBitmap().to_ulong();
    std::memcpy(ptr, &offsetbitmap, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    uint32_t offsets = p.rc.GetOffsets().to_ulong();
    std::memcpy(ptr, &offsets, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    uint32_t counters = p.rc.GetCounters();
    std::memcpy(ptr, &counters, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    std::memcpy(ptr, &p.seq_no, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    std::memcpy(ptr, &p.global_var, sizeof(uint32_t));

    return buffer;

}

Packet deserialize(const std::vector<char>& buffer, int rank)
{

    const char* ptr = buffer.data();

    uint32_t hlc, offsetbitmap, offsets, counters, seq_no, global_var;

    std::memcpy(&hlc, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&offsetbitmap, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&offsets, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&counters, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&seq_no, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&global_var, ptr, sizeof(int)); ptr += sizeof(int);

    ReplayClock rc = ReplayClock(hlc, rank, (std::bitset<NUM_PROCS>)offsetbitmap, (std::bitset<64>)offsets, counters, EPSILON, INTERVAL);

    Packet p = Packet(seq_no, rc, global_var);

    return p;
}

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
    ReplayClock rc = ReplayClock((uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() / INTERVAL, rank, EPSILON, INTERVAL);

    // Each process increments the global variable multiple times (N iterations)
    for (int i = 0; i < N; i++) {
        increment_global_variable(global_var, rank);

        auto now = std::chrono::system_clock::now();
        rc.SendLocal((uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() / INTERVAL);

        Packet p = Packet(i, rc, global_var);

        std::cout << "Process " << rank << " sending to " << rank + 1 << ": " << p.rc.GetHLC() << ", " << p.rc.GetBitmap() << ", " << p.rc.GetOffsets() << ", " << p.rc.GetCounters() << ", " << p.seq_no << ", " << p.global_var << std::endl;

        // Send updated value to the next process
        if (rank != size - 1) {

            std::vector<char> serialized_packet = serialize(p);
            MPI_Send(serialized_packet.data(), serialized_packet.size(), MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
            
        }

        // Receive updated value from the previous process
        if (rank != 0) {

            std::vector<char> deserialized_packet(sizeof(uint32_t)*6);
            MPI_Recv(deserialized_packet.data(), deserialized_packet.size(), MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Packet recv = deserialize(deserialized_packet, rank);

            std::cout << "Process " << rank << " received from " << rank - 1 << ": " << recv.rc.GetHLC() << ", " << recv.rc.GetBitmap() << ", " << recv.rc.GetOffsets() << ", " << recv.rc.GetCounters() << ", " << recv.seq_no << ", " << recv.global_var << std::endl;

            auto now = std::chrono::system_clock::now();
            rc.Recv(recv.getReplayClock(), (uint32_t)std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() / INTERVAL);
            global_var = recv.getGlobalVar();
        }
    }

    // At the end of each process, print the value of the global variable
    std::cout << "Process " << rank << " has final global variable value: " << global_var << std::endl;

    MPI_Finalize();
    return 0;
}
