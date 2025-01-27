#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "replay-clock.h"

#define N 4  // Size of the matrix and vector

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

void print_vector(int *vec, int size, int rank) {
    printf("Process %d received result: ", rank);
    for (int i = 0; i < size; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[N][N] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int x[N] = {1, 2, 3, 4};  // Input vector
    int local_result = 0;
    int global_result[N];

    if(rank == 0)
    {
        printf("TYPE,SENDER,RECIEVER,SEQNO,HLC,BITMAP,OFFSETS,COUNTERS\n");
    }

    if (size != N) {
        if (rank == 0) {
            printf("Please run with %d processes.\n", N);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }


    auto now = std::chrono::system_clock::now();
    ReplayClock rc = ReplayClock((uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL, rank, EPSILON, INTERVAL);

    // Each process computes a row of the matrix-vector multiplication
    for (int j = 0; j < N; j++) {
        local_result += A[rank][j] * x[j];
    }

    // Race condition: All processes try to send first, leading to potential deadlock
    for (int i = 0; i < size; i++) {
        if (rank != i) {
            
            auto now = std::chrono::system_clock::now();
            rc.SendLocal((uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL);
            Packet p = Packet(i, rc, local_result);
            std::vector<char> serialized_packet = serialize(p);
            printf("SEND,%d,%d,%d,%d,%s,%s,%d,%d\n", rank, rank + 1, p.seq_no, p.rc.GetHLC(), p.rc.GetBitmap().to_string().c_str(), p.rc.GetOffsets().to_string().c_str(), p.rc.GetCounters());
            MPI_Send(serialized_packet.data(), serialized_packet.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);


            std::vector<char> deserialized_packet(sizeof(uint32_t)*6);
            MPI_Recv(deserialized_packet.data(), deserialized_packet.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Packet recv = deserialize(deserialized_packet, rank);
            printf("RECV,%d,%d,%d,%d,%s,%s,%d,%d\n", rank, rank + 1, recv.seq_no, recv.rc.GetHLC(), recv.rc.GetBitmap().to_string().c_str(), recv.rc.GetOffsets().to_string().c_str(), recv.rc.GetCounters());
            auto now = std::chrono::system_clock::now();
            rc.Recv(recv.getReplayClock(), (uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL);
            global_result[i] = recv.global_var;



        } else {
            printf("LOCAL,%d,%d,%d,%d,%s,%s,%d,%d\n", rank, rank, 0, rc.GetHLC(), rc.GetBitmap().to_string().c_str(), rc.GetOffsets().to_string().c_str(), rc.GetCounters());
            global_result[i] = local_result;
        }
    }

    // Print the result vector from each process
    print_vector(global_result, N, rank);

    MPI_Finalize();
    return 0;
}
