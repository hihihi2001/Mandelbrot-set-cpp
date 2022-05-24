#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "DeviceProperties.cuh"
#include "Header.cuh"

#define NUMBER float

using std::cout;
using namespace std;

const int height = 1024;
const int width = 2048;

__global__ void mandPixel(BYTE* image, int log2limit, NUMBER infRadius, NUMBER bottomLeftX, NUMBER bottomLeftY, NUMBER dx, NUMBER dy);

int main()
{
	system("title Mandelbrot set by Kozak Aron");
// GPU selection
	ListCUDADevices();
	cudaSetDevice(0);
// allocation
    Measure Stopper = Measure();
    BYTE* image = new BYTE[3 * width * height];
    BYTE* gpu_image;
    cudaMalloc((void**)&gpu_image, 3 * height * width * sizeof(BYTE));
    char* imageFileName = (char*)"bitmapImage.bmp";
    int BlockSize = 32;
    int GridSize = height * width / BlockSize; // 65536
// create mandelbrot:
    cout << "start" << endl;
    Stopper.start();
    int trialNum = 32;

    NUMBER bottomLeftX(-3), bottomLeftY(-1),  sizeX(4),  sizeY(2);
    const NUMBER infRadius = 4.0;
    int log2limit = 8;
    const NUMBER dx = sizeX / width;
    const NUMBER dy = sizeY / height;

    for (int trial = 0; trial < trialNum; trial++)
    {
        mandPixel << <GridSize, BlockSize >> > (gpu_image, log2limit, infRadius, bottomLeftX, bottomLeftY, dx, dy);
        cudaDeviceSynchronize();
        cudaMemcpy(image, gpu_image, 3 * height * width * sizeof(BYTE), cudaMemcpyDeviceToHost);
    }

    long long int time = Stopper.stop(US);
    cout << "create mandelbrot: " << time / trialNum << " us" << endl;
// save mandelbrot:
    Stopper.start();
    generateBitmapImage(image, height, width, imageFileName);
    cout << "save bitmap image: " << Stopper.stop(MS) << " ms" << endl;
// free memory:
    delete[] image;
    cudaFree(gpu_image);
    return 0;
}

__global__ void mandPixel(BYTE* image, int log2limit, NUMBER infRadius, NUMBER bottomLeftX, NUMBER bottomLeftY, NUMBER dx, NUMBER dy)
{
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int c = threadID % width;
    int r = (threadID - c) / width;

    const unsigned int limit = (1 << log2limit) - 1;
    const NUMBER a_c = bottomLeftX + c * dx;
    const NUMBER b_c = bottomLeftY + r * dy;
    NUMBER a(0.0), b(0.0), temp;
    unsigned int iteration = 0;

    while ((++iteration < limit) && ((a * a + b * b) < infRadius))
    {
        temp = a * a - b * b + a_c;
        b = 2.0 * a * b + b_c;
        a = temp;
    }

    int pixel = 3 * threadID; //3 * (width * r + c);
    color(&image[pixel], iteration, log2limit);
}