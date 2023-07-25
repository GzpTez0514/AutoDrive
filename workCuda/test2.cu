#include <stdio.h>
#include <math.h>
// index data
// z[i] = x[i] + y[i]
// memory allocation
// memory copy
// kernel func
// memcopy copy

__global__ void vecAdd(const double *x, const double *y, double *z, int count)
{   
    /*
    blockDim.x: 每个块在x方向上的线程数量，这是块的尺寸
    blockIdx.x: 当前线程所在的块在x方向上的索引，值从0开始，一直到尺寸(gridDim.x) - 1
    threadIdx.x: 当前线程在其所在的块中在x方向的索引，值从0开始，一直到尺寸(blockDim.x) - 1

    blockDim.x * blockIdx.x + threadIdx.x: 把二维的块和线程索引转换成一个一维的全局索引
    */
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    // 
    if(index < count) 
    {
        z[index] = x[index] + y[index];
    }
}

void vecAdd_cpu(const double *x, const double *y, double *z, int count)
{
    for(int i = 0; i < count; ++i)
    {
        z[i] = x[i] + y[i];

    }
}

int main()
{
    const int N = 1000;

    const int M = sizeof(double) * N;  // 内存大小

    // cpu memory alloc
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);
    double *result_cpu = (double*) malloc(M);

    // 值初始化
    for(int i = 0; i < N; ++i)
    {
        h_x[i] = 1;
        h_y[i] = 2;
    }

    // GPU分配空间
    double *d_x, *d_y, *d_z;
    cudaMalloc((void**) &d_x, M);
    cudaMalloc((void**) &d_y, M);
    cudaMalloc((void**) &d_z, M);

    // cpu上初始化好的数据传输给GPU
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // 定义核函数进行进行计算
    const int block_size = 128;  // 每个block里面有128个线程
    const int grid_size = (N + block_size - 1) / block_size;

    vecAdd<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    cudaMemcpy(h_z, d_z, M, cudaMemcpyHostToDevice);

    vecAdd_cpu(h_x, h_y, result_cpu, N);
    bool error = false;
    for(int i = 0; i < N; ++i)
    {
        if(fabs(result_cpu[i] - h_z[i]) > (1.0e-10))
        {
            error = true;
        }
    }
    printf("Result: %s \n", error ? "Error" : "Pass");

    free(h_x);
    free(h_y);
    free(h_z);
    free(result_cpu);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

}

