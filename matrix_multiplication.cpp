#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>
#include <time.h>
#include <unistd.h>
#include "replay-clock.h"

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

    std::memcpy(&hlc, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    std::memcpy(&offsetbitmap, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    std::memcpy(&offsets, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    std::memcpy(&counters, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    std::memcpy(&seq_no, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    std::memcpy(&global_var, ptr, sizeof(uint32_t)); ptr += sizeof(uint32_t);

    ReplayClock rc = ReplayClock(hlc, rank, (std::bitset<NUM_PROCS>)offsetbitmap, (std::bitset<64>)offsets, counters, EPSILON, INTERVAL);

    Packet p = Packet(seq_no, rc, global_var);

    return p;
}

void print_vector(int *vec, int size, int rank, const char *label) {
    printf("Process %d %s: ", rank, label);
    for (int i = 0; i < size; i++) {
        printf("%d ", vec[i]);
    }
    printf("\n");
}

// Function to generate a random NxN matrix
void generate_random_matrix(int A[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;  // Random values between 0-9
        }
    }
}

// Function to generate a random vector of size N
void generate_random_vector(int X[N]) {
    for (int i = 0; i < N; i++) {
        X[i] = rand() % 10;  // Random values between 0-9
    }
}

// Function to broadcast the matrix to all processes
void broadcast_matrix(int A[N][N]) {
    for (int i = 0; i < N; i++) {
        MPI_Bcast(A[i], N, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

// Function to broadcast the vector to all processes
void broadcast_vector(int X[N]) {
    MPI_Bcast(X, N, MPI_INT, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[N][N];
    int x[N];  // Input vector

    
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

    // Process 0 initializes the random matrix and vector and broadcasts them
    if (rank == 0) {
        srand(time(NULL));  // Seed for random number generation
        generate_random_matrix(A);
        generate_random_vector(x);

        printf("Generated matrix A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%d ", A[i][j]);
            }
            printf("\n");
        }

        print_vector(x, N, rank, "generated vector X");
    }

    // Broadcast the matrix and vector to all processes
    broadcast_matrix(A);
    broadcast_vector(x);


    auto now = std::chrono::system_clock::now();
    ReplayClock rc = ReplayClock((uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL, rank, EPSILON, INTERVAL);

    // Each process computes a row of the matrix-vector multiplication
    for (int j = 0; j < N; j++) {
        local_result += A[rank][j] * x[j];
    }

    // Force race condition with random delays
    srand(time(NULL) + rank);  // Different seed per process
    usleep((rand() % 1000) * 10000);  // Random sleep between 0-1 seconds

    // Race condition: All processes try to send first, leading to potential deadlock
    for (int i = 0; i < size; i++) {
        if (rank != i) {
            
            auto now = std::chrono::system_clock::now();
            rc.SendLocal((uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL);
            Packet p = Packet(i, rc, local_result);
            std::vector<char> serialized_packet = serialize(p);
            printf("SEND,%d,%d,%d,%d,%s,%s,%d,SentLocalResult\n", rank, rank + 1, p.seq_no, p.rc.GetHLC(), p.rc.GetBitmap().to_string().c_str(), p.rc.GetOffsets().to_string().c_str(), p.rc.GetCounters());
            MPI_Send(serialized_packet.data(), serialized_packet.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD);


            std::vector<char> deserialized_packet(sizeof(uint32_t)*6);
            MPI_Recv(deserialized_packet.data(), deserialized_packet.size(), MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            Packet recv = deserialize(deserialized_packet, rank);
            printf("RECV,%d,%d,%d,%d,%s,%s,%d,ReceivedLocalResult\n", rank, rank + 1, recv.seq_no, recv.rc.GetHLC(), recv.rc.GetBitmap().to_string().c_str(), recv.rc.GetOffsets().to_string().c_str(), recv.rc.GetCounters());
            now = std::chrono::system_clock::now();
            rc.Recv(recv.getReplayClock(), (uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL);
            global_result[i] = recv.global_var;



        } else {
            auto now = std::chrono::system_clock::now();
            rc.SendLocal((uint32_t)std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count() / INTERVAL);
            printf("LOCAL,%d,%d,%d,%d,%s,%s,%d,UpdatingGlobalResult\n", rank, rank, 0, rc.GetHLC(), rc.GetBitmap().to_string().c_str(), rc.GetOffsets().to_string().c_str(), rc.GetCounters());
            global_result[i] = local_result;
        }
    }

    // Print the computed result
    print_vector(global_result, N, rank, "result vector");

    MPI_Finalize();
    return 0;
}
