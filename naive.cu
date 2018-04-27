#include <iostream>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
using namespace std;

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i+=stride)
    y[i] = x[i] + y[i];
}
void testTrush(){
    // H has storage for 4 integers
    thrust::host_vector<float*> H(2);

    // initialize individual elements

    // print contents of H
    for(int i = 0; i < H.size(); i++){
        H[i] = (float*)malloc(2*sizeof(float));
        H[i][0] = 0;H[i][1] = 1;
        std::cout << "H[" << i << "] = " << H[i][0] <<"--"<<H[i][1]<< std::endl;
    }
    // float test[3];
    // *test = {.0f, .0f, .0f};
    // for(int i=0;i<3;i++)
    //     cout<<test[i]<<endl;
    //
    // // resize H
    // H.resize(2);
    //
    // std::cout << "H now has size " << H.size() << std::endl;
    //
    // // Copy host_vector H to device_vector D
    // thrust::device_vector<int> D = H;
    //
    // // elements of D can be modified
    // D[0] = 99;
    // D[1] = 88;
    //
    // // print contents of D
    // for(int i = 0; i < D.size(); i++)
    //     std::cout << "D[" << i << "] = " << D[i] << std::endl;
}
int main()
{
    testTrush();
  int N = 1<<20;
  float *x, *y;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  int blockSize = 256;
  int numOfBlocks = (N+blockSize-1)/blockSize;
  // Run kernel on 1M elements on the GPU
  add<<<numOfBlocks, blockSize>>>(N, x, y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}
