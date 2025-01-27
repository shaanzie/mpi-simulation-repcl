mpic++ -std=c++11 matrix_multiplication.cpp replay-clock.cpp

mpiexec -n 10 -hostfile nodes ./a.out