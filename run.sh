mpic++ -std=c++11 matrix_multiplication.cpp replay-clock.cpp

mpiexec -n 4 -hostfile nodes ./a.out