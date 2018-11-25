from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = np.arange(100, dtype='i')
else:
    data = np.empty(100, dtype='i')
print "data on rank: ", rank
print data
comm.Bcast(data, root=0)
print "after bcast on rank ", rank
print data
for i in range(100):
    assert data[i] == i
