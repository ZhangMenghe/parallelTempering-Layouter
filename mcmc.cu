#include <iostream>
#include <assert.h>
#include <limits.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>

using namespace std;

const unsigned int nBlocks = 10 ;
const unsigned int WHICH_GPU = 0;
int seed;

__device__ float density_function(float beta, float cost) {
    // printf("%f-%f\n", beta, cost);
	return exp2f(-beta * cost);
}

__device__ float get_randomNum(unsigned int seed, int maxLimit) {
  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t state;

  /* we have to initialize the state */
  curand_init(seed, /* the seed controls the sequence of random values that are produced */
              0, /* the sequence number is only important with multiple cores */
              0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
              &state);

  /* curand works like rand - except that it takes a state as a parameter */
  return curand(&state) % maxLimit;
 // int res = curand(&state) % maxLimit;
 // printf("%d ", res);
 // return res;
}



__device__ float cost_function(float * data, int length){
    //dummy cost, just sum up all
    float res = 0;

    for(int i=0; i<length; i++)
        res += data[i];
    // printf("res: %f\n", res);
    return res/1000;
}
__device__
void changeTemparature(float * temparature, unsigned int seed){
    int t1 = get_randomNum(seed, nBlocks);
    int t2=t1;
    while(t2 == t1)
        t2 = get_randomNum(seed + 100, nBlocks);
    float tmp = temparature[t1];
    temparature[t1] = temparature[t2];
    temparature[t2] = tmp;
}
__device__
void ActualHW(int randTimes, int numofObjs, unsigned int seed, int* pickedIdAddr, float*sArray, float * cost, float *temparature){
    // bool hit = false;
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    for(int t=0; t<randTimes; t++){
        if(pickedIdAddr[t] == threadIdx.x){
            if(t % 10 == 0)
                changeTemparature(temparature, seed+index);
            float cost_pri = cost_function(sArray, numofObjs);
            float p0 = density_function(temparature[blockIdx.x], cost_pri);
            float tmpKeep = sArray[threadIdx.x];
            sArray[threadIdx.x] = get_randomNum(seed+index, 1000);

            float cost_post = cost_function(sArray, numofObjs);
            float p = density_function(temparature[blockIdx.x], cost_post);
            float alpha = min(1.0f, p/p0);
            // printf("p/p0: %f\n", p/p0);
            float t =0.8f;
            //change back
            if(alpha>t)
                sArray[threadIdx.x] = tmpKeep;
            else{
                if(sArray[threadIdx.x]>tmpKeep)
                    printf("%f - %f\n", tmpKeep, sArray[threadIdx.x]);
                cost[blockIdx.x] = cost_post;
            }


            // hit = true;
        }
    }
    // return hit;
}
__global__
void simpleHW(int numofObjs, float * gValues, float* gArray,unsigned int seed,int*pickedIdxs, int randTimes){
    //here should be dynamic shared mem
    //__shared__ float sArray[30];
    extern __shared__ float sharedMem[];
    float * sArray = sharedMem;
    float * lastSumUp = (float *) & sArray[nBlocks*numofObjs];
    float * temparature = (float *) & lastSumUp[nBlocks];
    //initialize
    int startIdx = blockIdx.x * numofObjs;
    int idx =  startIdx+ threadIdx.x;

    sArray[idx] = gValues[threadIdx.x];
    temparature[blockIdx.x] = -get_randomNum(seed+blockIdx.x, 100) / 10;
    // printf("temp: %f", temparature[blockIdx.x]);
    lastSumUp[blockIdx.x] = 0;
    for(int i = 0;i<numofObjs; i++)
        lastSumUp[blockIdx.x] += gValues[i];

    int* pickedIdAddr = &pickedIdxs[blockIdx.x * randTimes];

    ActualHW(randTimes, numofObjs, seed, pickedIdAddr, &sArray[startIdx], lastSumUp, temparature);
    __syncthreads();
    gArray[idx] = sArray[idx];
}

void generate_suggestions(){
    float *gValues;
    float * gArray;
    int * pickedIdxs;

    int numofObjs = 5;

    int nTimes =20000;

    int totalSize = nBlocks*numofObjs* sizeof(float);

    cudaMallocManaged(&gValues, numofObjs * sizeof(float));
    for(int i=0; i<numofObjs; i++)
        gValues[i] = 1000;
    cudaMallocManaged(&pickedIdxs, nBlocks*nTimes * sizeof(int));
    for(int i=0; i<nBlocks*nTimes; i++)
        pickedIdxs[i] = rand()%numofObjs;
    // for(int i=0; i<nBlocks*nTimes; i++)
    //     cout<<pickedIdxs[i]<<" ";
    // cout<<endl;

    cudaMallocManaged(&gArray, totalSize);
    //dynamic shared mem, <<<nb, nt, sm>>>
    simpleHW<<<nBlocks, numofObjs, totalSize + 2*nBlocks*sizeof(float)>>>(numofObjs, gValues, gArray,time(NULL),pickedIdxs,nTimes);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    for(int i=0;i<nBlocks;i++){
        for(int j=0; j<numofObjs; j++)
            cout<<gArray[i * numofObjs+ j]<<" ";
        cout<<endl;
    }

    // Free memory
    cudaFree(gValues);
    cudaFree(gArray);
    cudaFree(pickedIdxs);
}
void setUpDevices(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if(WHICH_GPU <= deviceCount) {
    cudaError_t err = cudaSetDevice(WHICH_GPU);
    if(err != cudaSuccess)
        cout<< "CUDA error:" <<cudaGetErrorString(err)<<endl;
    }
    else {
        cout << "Invalid GPU device " << WHICH_GPU << endl;
        exit(-1);
    }
    int wgpu;
    cudaGetDevice(&wgpu);
    cudaDeviceReset();
}

// int main(int argc, char** argv){
//     setUpDevices();
//     seed = time(NULL);
//     srand(seed);
//     generate_suggestions();
//     return 0;
// }
