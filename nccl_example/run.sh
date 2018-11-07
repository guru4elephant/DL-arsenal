for i in 1 10 100
do
    for j in 1024 1048576 33554432 67108864 134217728 268435456 536870912
    do
        rm -rf /tmp/.nvprof
        nvprof --profile-child-processes /home/users/dongdaxiang/.ndt/software/mpi/bin/mpirun -np 8 hehe 1024 $i $j 2> nvprof.$i.$j
    done
done
