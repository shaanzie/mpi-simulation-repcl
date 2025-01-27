NUM_PROCS=$1
EPSILON=$2
INTERVAL=$3
MAX_OFFSET_SIZE=$7

# Function to calculate the number of bits required to store an integer
num_bits() {
    number=$1
    bits=0
    while ((number > 0)); do
        ((bits++))
        number=$((number >> 1))
    done
    echo $bits
}

rm replay-config.h
echo "#define REPCL_CONFIG_H" >> replay-config.h
echo "#define NUM_PROCS $NUM_PROCS" >> replay-config.h
echo "#define EPSILON $EPSILON" >> replay-config.h
echo "#define INTERVAL $INTERVAL" >> replay-config.h
echo "#define MAX_OFFSET_SIZE $(num_bits $EPSILON)" >> replay-config.h
echo "#define N $NUM_PROCS" >> replay-config.h

mpic++ -std=c++11 matrix_multiplication.cpp replay-clock.cpp

mpiexec -n $NUM_PROCS -hostfile nodes ./a.out