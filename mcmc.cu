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
extern __shared__ float sFloats[];
__device__ __managed__ Room* room;

class automatedLayout
{
private:
	layoutConstrains *constrains;
	float *weights;
	int debugParam = 0;
    void random_along_wall(int furnitureID);


public:
    // Room * room;
	float min_cost;
    float *resTransAndRot;
	automatedLayout(vector<float>in_weights);
	void generate_suggestions();
	void display_suggestions();
	void initial_assignment(const Room* refRoom);
	__device__ float cost_function();
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
	layout->initial_assignment(m_room);
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
void Metropolis_Hastings(int* pickedIdAddr, float* costList, float* temparature){

}
__global__
void Do_Metropolis_Hastings(automatedLayout* al, int * pickedIdxs,unsigned int seed){
	float* temparature = sFloats;
	float* costList = (float *) & temparature[nBlocks * sizeof(float)];
	temparature[blockIdx.x] = -get_randomNum(seed+blockIdx.x, 100) / 10;
	costList[blockIdx.x] = al->cost_function();
	int* pickedIdAddr = &pickedIdxs[blockIdx.x * nTimes];
	Metropolis_Hastings(pickedIdAddr, costList, temparature);
	__syncthreads();

}
__global__
void AssignFurnitures(){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	sObjs[index] = room->deviceObjs[threadIdx.x];
	__syncthreads();
}


void Room::RoomCopy(const Room & m_room){
	objctNum = m_room.objctNum;
	freeObjNum = m_room.freeObjNum;
	half_width = m_room.half_width;
	half_height = m_room.half_height;
	indepenFurArea = m_room.indepenFurArea;
	obstacleArea = m_room.obstacleArea;
	wallArea = m_room.wallArea;
	overlappingThreshold = m_room.overlappingThreshold;
	colCount = m_room.colCount;
	rowCount = m_room.rowCount;
	cudaMemcpy(freeObjIds, m_room.freeObjIds, freeObjNum* sizeof(int), cudaMemcpyHostToDevice);
	cudaMallocManaged(&deviceObjs,  objctNum * sizeof(singleObj));
	for(int i=0; i<objctNum; i++)
		deviceObjs[i] = m_room.objects[i];


	int tMem = colCount*rowCount * sizeof(unsigned char);
	cout<<colCount<<"-"<<rowCount<<endl;
	cudaMallocManaged(&furnitureMask, tMem);
	cudaMallocManaged(&furnitureMask_initial, tMem);
	cudaMemcpy(furnitureMask, m_room.furnitureMask, tMem, cudaMemcpyHostToDevice);
	cudaMemcpy(furnitureMask_initial, m_room.furnitureMask_initial, tMem, cudaMemcpyHostToDevice);
	//TODO:map..obstacle
	// cout<<"test- "<<int(furnitureMask[100])<<endl;

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

	AssignFurnitures<<<nBlocks, room->objctNum, objMem >>>();
	cudaDeviceSynchronize();
	Do_Metropolis_Hastings<<<nBlocks, room->objctNum, temMem>>>(this, pickedIdxs, time(NULL));
	cudaDeviceSynchronize();

	cudaFree(resTransAndRot);
	cudaFree(pickedIdxs);
	cudaFree(weights);
	cudaFree(room);


    // for(int i=0;i<nBlocks;i++){
    //     // for(int j=0; j<numofObjs; j++)
    //         cout<<rArray[i]<<" ";
    //     cout<<endl;
    // }

}
void automatedLayout::random_along_wall(int furnitureID){

}
__device__
float automatedLayout::cost_function(){
	return 0;
}
void automatedLayout::initial_assignment(const Room * refRoom){
	for (int i = 0; i < refRoom->freeObjNum; i++) {
		singleObj* obj = &room->deviceObjs[refRoom->freeObjIds[i]];
		if (obj->adjoinWall)
			random_along_wall(refRoom->freeObjIds[i]);
		else if (obj->alignedTheWall)
			cout<<"do nothing now"<<endl;
			// room->set_obj_zrotation(refRoom->walls[rand() % refRoom->wallNum].zrotation, room->freeObjIds[i]);
	}
	room->update_furniture_mask();
}

// int main(int argc, char** argv){
//     setUpDevices();
//     seed = time(NULL);
//     srand(seed);
//     generate_suggestions();
//     return 0;
// }
