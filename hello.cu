#include<stdio.h>
#include<cuda_runtime.h>
__global__ void helloFromGPU(void){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world form idx %d, block %d, thread %d\n", idx, blockIdx.x, threadIdx.x);

}
int main(void){
int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        printf("There is no device supporting CUDA\n");
        return 0;
    }
    int dev;
    for(dev=0; dev<deviceCount; dev++){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("Device %d: %s\n", dev, deviceProp.name);
    }
    cudaSetDevice(0);
    int nElm = 100000;
    dim3 blockSize (nElm/5);
    dim3 gridSize ((nElm+blockSize.x-1)/blockSize.x);
    //blockSize.x = nElm/5;
    //gridSize.x = (nElm+blockSize.x-1)/blockSize.x;
    helloFromGPU<<<gridSize,blockSize>>>();
    cudaDeviceReset();
    return 0;
}