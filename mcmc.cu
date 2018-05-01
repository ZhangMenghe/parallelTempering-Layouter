#include <limits.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include "layoutConstrains.h"
#include "room.h"
#include "predefinedConstrains.h"
#define RES_NUM 1
using namespace std;
using namespace cv;

const unsigned int nBlocks = 10 ;
const unsigned int WHICH_GPU = 0;
const unsigned int nTimes =20;

void roomInitialization(Room* m_room);

extern __shared__ singleObj sObjs[];
extern __shared__ float gtemparing[];
__device__ __managed__ Room* room;

class automatedLayout
{
private:
	layoutConstrains *constrains;
	float *weights;
	int debugParam = 0;
    void random_along_wall(int furnitureID);
    float cost_function();

public:
    // Room * room;
	float min_cost;
    float *resTransAndRot;
	automatedLayout(vector<float>in_weights);
	void generate_suggestions();
	void display_suggestions();
	__device__ void debugDevice(){
		// printf("id: %d\n", sObjs[0].id);
		// printf("res:\n", resTransAndRot[0]);
		// sObjs[0] = room->deviceObjs[1];
		// debugParam = 2;//room->deviceObjs[1].id;
	}
	__device__ void assignId(){
		printf("%d\n", room->deviceObjs[0].id );
		// sObjs[0] = room->deviceObjs[0];
		// sObjs[0].id = 9;
		// resTransAndRot[0] = 0;
		// printf("%f", resTransAndRot[0]);
	}
};

int seed;
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
void startToProcess(Room * m_room, vector<float> weights){
	//cout<<"hello from mcmc"<<endl;
	setUpDevices();
	roomInitialization(m_room);
	automatedLayout * layout = new automatedLayout(weights);
	layout->generate_suggestions();
   // 	// layout->display_suggestions();
}

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



__device__ float cost_function_device(float * data, int length){
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
            float cost_pri = cost_function_device(sArray, numofObjs);
            float p0 = density_function(temparature[blockIdx.x], cost_pri);
            float tmpKeep = sArray[threadIdx.x];
            sArray[threadIdx.x] = get_randomNum(seed+index, 1000);

            float cost_post = cost_function_device(sArray, numofObjs);
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
void Do_Metropolis_Hastings(unsigned int seed, float * rArray){
	gtemparing[blockIdx.x] = blockIdx.x;//-get_randomNum(seed+blockIdx.x, 100) / 10;
	__syncthreads();
	rArray[blockIdx.x] = gtemparing[blockIdx.x];
}
__global__
void AssignFurnitures(int * pickedIdxs, unsigned int seed){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	sObjs[index] = room->deviceObjs[threadIdx.x];
	__syncthreads();
	printf("%d\n", sObjs[index].id);
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
void naiveCUDA(){
	float *gValues;
    float * gArray;
    int * pickedIdxs;

    int numofObjs = 5;

    // int nTimes =20000;

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
void Room::RoomCopy(const Room & m_room){
	objctNum = m_room.objctNum;
	cudaMallocManaged(&deviceObjs,  objctNum * sizeof(singleObj));
	for(int i=0; i<objctNum; i++)
		deviceObjs[i] = m_room.objects[i];

}
void roomInitialization(Room* m_room){
	cudaMallocManaged(&room,  sizeof(Room));
	room->RoomCopy(*m_room);
}

automatedLayout::automatedLayout(vector<float>in_weights) {

	// constrains = new layoutConstrains(m_room);
	min_cost = INFINITY;

	// cout<<"shenmemaobing: " <<room->deviceObjs[1].id<<endl;
	cudaMallocManaged(&weights, in_weights.size() * sizeof(float));
	for(int i=0;i<in_weights.size();i++)
		weights[i] = in_weights[i];

	cudaMallocManaged(&resTransAndRot,  4 * sizeof(float));
	// for(int i=0;i<RES_NUM*4;i++)
	// 	resTransAndRot[i] = i;
	// float tmpf[] = {1.0f, 1.5f, 0.5f,1.0f};
	// cudaMemcpy(resTransAndRot, tmpf, 4*sizeof(float), cudaMemcpyHostToDevice);
}


void automatedLayout:: generate_suggestions(){
	if(room->objctNum == 0)
		return;
    int * pickedIdxs; //should be in global mem


    cudaMallocManaged(&pickedIdxs, nBlocks * nTimes * sizeof(int));
    for(int i=0; i<nBlocks*nTimes; i++)
        pickedIdxs[i] = rand()%room->objctNum;

	//memory to store result, should be in global mem


    //dynamic shared mem, <<<nb, nt, sm>>>
	//obj +  temparing + room + weight
	// int sharedMem = nBlocks * (sizeof(room->objects)+ 2*sizeof(float)+ sizeof(*room) + sizeof(*deviceWeights));
	int objMem = nBlocks * room->objctNum * sizeof(singleObj);
	int temMem = nBlocks * sizeof(float);
	float * rArray;
	cudaMallocManaged(&rArray, nBlocks * sizeof(float));

	AssignFurnitures<<<nBlocks, room->objctNum, objMem >>>(pickedIdxs, time(NULL));
	cudaDeviceSynchronize();
	Do_Metropolis_Hastings<<<nBlocks, room->objctNum, temMem>>>(time(NULL), rArray);
	cudaDeviceSynchronize();

	cudaFree(resTransAndRot);
	cudaFree(pickedIdxs);
	cudaFree(weights);
	cudaFree(room);


    for(int i=0;i<nBlocks;i++){
        // for(int j=0; j<numofObjs; j++)
            cout<<rArray[i]<<" ";
        cout<<endl;
    }

}
void automatedLayout::random_along_wall(int furnitureID){

}
float automatedLayout::cost_function(){
	return 0;
}

// void automatedLayout::initial_assignment(){
// 	for (int i = 0; i < room->freeObjIds.size(); i++) {
// 		singleObj* obj = &room->objects[room->freeObjIds[i]];
// 		if (obj->adjoinWall)
// 			random_along_wall(room->freeObjIds[i]);
// 		else if (obj->alignedTheWall)
// 			room->set_obj_zrotation(room->walls[rand() % room->wallNum].zrotation, room->freeObjIds[i]);
// 	}
// 	room->update_furniture_mask();
// 	min_cost = cost_function();
// 	if (min_cost == -1)
// 		min_cost = INFINITY;
//
// }
// int main(int argc, char** argv){
//     setUpDevices();
//     seed = time(NULL);
//     srand(seed);
//     generate_suggestions();
//     return 0;
// }
