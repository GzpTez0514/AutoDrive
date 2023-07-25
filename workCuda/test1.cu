#include <stdio.h>

__global__ void my_first_kernel()
{   
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    printf("Hello world from thread(thread index:(%d, %d), block index:(%d, %d))! \n", tidy, tidx, bidy, bidx);
}

// thread --> block --> grid
// SM stream multi-processor  流多处理器
// total threads: block_size * grid_size
int main()
{
    printf("Hello world from CPU \n");

    // 一维
    // int block_size = 3;
    // int grid_size = 2;

    // 二维
    dim3 block_size(3, 3);
    // t00, t01, t02
    // t10, t11, t12
    // t20, t21. t22
    dim3 grid_size(2, 2);
    // b00, b01
    // b10, b11

    my_first_kernel<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();  // 告诉cpu gpu上的函数执行完毕了

    return 0; 
}