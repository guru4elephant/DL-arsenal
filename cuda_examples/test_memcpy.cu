
int main()
{
    const int blockSize = 256, nStreams = 4;
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    cudaMalloc((int**)&d_a, bytes);

    memset(h_a, 0, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);

    return 0;
}

int main()
{
    const int blockSize = 256, nStreams = 4;
    const int n = 4 * 1024 * blockSize;
    const int streamSize = n / nStreams;
    const int streamBytes = streamSize * sizeof(float);
    const int bytes = n * sizeof(float);
    float *a, *d_a;
    
    cudaStream_t stream[nStream];
    for (int i = 0; i < nStreams; ++i) {
        checkCuda(cudaStreamCreate(&stream[i]));
    }
    memset(a, 0, bytes);
    

    checkCuda(cudaMallocHost((void**)&a, bytes));
    checkCuda(cudaMalloc((void**)&d_a, bytes));
    for (int i = 0; i < nStreams; ++i) {
        checkCuda(cudaMemcpyAsync());
        checkCuda()
    }
}