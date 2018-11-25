from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
local_hostname = socket.gethostname()
recv_obj = comm.allgather(local_hostname)
print "ranks on rank ", rank
print recv_obj

local_rank = 0
for i in range(size):
    if i == rank:
        break
    else:
        if local_hostname == recv_obj[i]:
            local_rank += 1
print "rank ", rank
print "local rank ", local_rank
nccluniqueId = [-1]
if rank == 0:
    nccluniqueId = [1234]
nccluniqueId = comm.bcast(nccluniqueId, root=0)
print "ncclid on rank ", rank
print nccluniqueId
