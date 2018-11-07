#include <stdio.h>
#include "nccl.h"
#include "mpi.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define LOGERR(format, args...) (fprintf(stderr, "[%s:%d:%s] " format "\n",\
                                         __FILE__, __LINE__, __FUNCTION__, ##args))

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static void getHostName(char *hostname, int maxlen) {
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++) {
        if (hostname[i] == '.') {
            hostname[i] = '\0';
            return;
        }
    }
}

static uint64_t getHostHash(const char *string) {
    // Based on DJB2, result = result * 33 + char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++) {
        result = ((result << 5) + result) + string[c];
    }
    return result;
}

static __thread cublasHandle_t _g_gpu_handle;
static __thread bool _g_init_handle = false;
void local_gemm(const int M, const int N, const int K,
                const float alpha, const float * A, const float * B, const float beta,
                float * C) {
    cublasOperation_t cuTransA = CUBLAS_OP_N;
    cublasOperation_t cuTransB = CUBLAS_OP_N;
    if (false == _g_init_handle) {
        cublasCreate(&_g_gpu_handle);
        _g_init_handle = true;
    }
    cublasSgemm(_g_gpu_handle, cuTransB, cuTransA,
                N, M, K, &alpha, B,M, A, K, &beta, C, N);
}


// ncclAllReduce
// ncclReduce

int main(int argc, char *argv[])
{
    if (argc != 4) {
        LOGERR("%s size matmul_num send_buf_size", argv[0]);
        exit(-1);
    }
    int size = atoi(argv[1]); // 1024
    int mat_num = atoi(argv[2]); // 1, 10, 100
    int send_size = atoi(argv[3]); // 1024, 1024 * 1024, 1024 * 1024 * 1024
    LOGERR("size: %d, mat_num: %d, send_size: %d", size, mat_num, send_size);
    int mat_size = size * size; 
    int my_global_rank = 0;
    int ranks = 0;
    int local_rank = 0;
    
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &my_global_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    
    // hash address of each rank as a uint64
    uint64_t host_hash[ranks];
    char hostname[1024];
    getHostName(hostname, 1024);
    host_hash[my_global_rank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                           host_hash, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    // init nccl 
    for (int i = 0; i < ranks; ++i) {
        if (i == my_global_rank) {
            break;
        }
        if (host_hash[i] == host_hash[my_global_rank]) {
            local_rank++;
        }
    }

    ncclUniqueId id;
    ncclComm_t comm;
    float * send_buff;
    float * recv_buff;
    float * A;
    float * B;
    float * C;
    cudaStream_t s;
    // prepare computation resources
    if (my_global_rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, 
                       MPI_COMM_WORLD));
    LOGERR("local rank: %d", local_rank);
    CUDACHECK(cudaSetDevice(local_rank));
    CUDACHECK(cudaMalloc(&send_buff, send_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv_buff, send_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&A, mat_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&B, mat_size * sizeof(float)));
    CUDACHECK(cudaMalloc(&C, mat_size * sizeof(float)));

    CUDACHECK(cudaStreamCreate(&s));

    NCCLCHECK(ncclCommInitRank(&comm, ranks, id, my_global_rank));
    // for loop to do computation and communication alternatively
    for (int i = 0; i < 1000; ++i) {
        // do computation first
        for (int j = 0; j < mat_num; ++j) {
            local_gemm(size, size, size, 1.0, A, B, 1.0, C);
        }
        // call nccl ALL Reduce here
        NCCLCHECK(ncclAllReduce((const void *)send_buff, 
                                (void *)recv_buff, 
                                send_size,
                                ncclFloat, ncclSum, comm, s));
        CUDACHECK(cudaStreamSynchronize(s));
    }
    
    CUDACHECK(cudaFree(send_buff));
    CUDACHECK(cudaFree(recv_buff));
    ncclCommDestroy(comm);
    MPICHECK(MPI_Finalize());
    LOGERR("Job success");
    return 0;
}
