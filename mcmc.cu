#include <iostream>
#include <string>
#include<fstream>
#include <limits.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
// #include "predefinedConstrains.h"
#include "room.cuh"
#include <time.h>
#define RES_NUM 1
#define THREADHOLD_T 0.8
using namespace std;
// using namespace cv;

const unsigned int nBlocks = 3;
const unsigned int nThreads = 16;
const unsigned int WHICH_GPU = 0;
const unsigned int nTimes =1;

void generate_suggestions();
extern __shared__ singleObj sObjs[];
extern __shared__ float sFloats[];

__device__ __managed__ Room* room;
__device__ __managed__ float weights[11]={1.0f};
__device__ __managed__ float resTransAndRot[RES_NUM * 4];

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
void debugCaller(){
    room->set_obj_zrotation(&room->deviceObjs[0], PI);
    room->set_obj_translation(&room->deviceObjs[0], -50, 0);
    // room->get_nearest_wall_dist(&room->deviceObjs[0]);
}
void startToProcess(Room * m_room){
	setUpDevices();
    cudaMallocManaged(&room,  sizeof(Room));
    *room = Room(*m_room);

    //debugCaller();
    clock_t start, finish;
    float costtime;
    start = clock();

	generate_suggestions();

    finish = clock();
    costtime = (float)(finish - start) / CLOCKS_PER_SEC;
    cout<<"Runtime: "<<costtime<<endl;
   // 	// layout->display_suggestions();
}

__device__
float density_function(float beta, float cost) {
    // printf("%f-%f\n", beta, cost);
	return exp2f(-beta * cost);
}

__device__
float get_randomNum(unsigned int seed, int maxLimit) {
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


__device__
void changeTemparature(float * temparature, unsigned int seed){
    int t1 = get_randomNum(seed, nBlocks);
    int t2 = t1;
    while(t2 == t1)
        t2 = get_randomNum(seed + 100, nBlocks);
    float tmp = temparature[t1];
    temparature[t1] = temparature[t2];
    temparature[t2] = tmp;
}
__device__
void randomly_perturb(){

}
__device__
void restoreOrigin(){

}
__device__
void getTemporalTransAndRot(){

}

__device__
float getWeightedCost(float* costList, int consStartId){
    costList[threadIdx.x] = threadIdx.x;//should be get constrainFactor from room;
    __syncthreads();
    float res = 0;
    for(int i=0; i<WEIGHT_NUM; i++)
        res += weights[i] * costList[consStartId + i];
    return res;
}
__device__
void Metropolis_Hastings(int* pickedIdAddr, float* costList, float* temparature, unsigned int seed){
    float cpost, p0, p1, alpha;
    int startId = blockIdx.x * nThreads;
    int index = startId + threadIdx.x;
    costList[index] = 0;
    float cpre = getWeightedCost(&costList[startId], room->objctNum);
    //first thread cost is the best cost of block
    costList[startId] = cpre;
    for(int nt = 0; nt<nTimes; nt++){
        if(pickedIdAddr[nt] == threadIdx.x){
            if(nt % 10 == 0)
                changeTemparature(temparature, seed+blockIdx.x);
            p0 = density_function(temparature[blockIdx.x], cpre);
            randomly_perturb(/*original keep sth to restore*/);
        }
        __syncthreads();
        cpost = getWeightedCost(costList, startId);
        costList[index] = 0;
        //per block operation
        if(pickedIdAddr[nt] == threadIdx.x){
            p1 = density_function(temparature[blockIdx.x], cpost);
            alpha = fminf(1.0f, p1/p0);
            if(alpha > THREADHOLD_T)
                restoreOrigin();
            else if(cpost < costList[blockIdx.x]){
                getTemporalTransAndRot();
                costList[startId] = cpost;
                cpre = cpost;
            }
        }
    }
}

__global__
void Do_Metropolis_Hastings(int * pickedIdxs, unsigned int seed){
	float* costList = sFloats;
    float* temparature = (float *) & costList[nBlocks * nThreads];

	temparature[blockIdx.x] = -get_randomNum(seed+blockIdx.x, 100) / 10;
    int* pickedIdAddr = &pickedIdxs[blockIdx.x * nTimes];
    Metropolis_Hastings(pickedIdAddr, costList, temparature, seed);
	__syncthreads();

}

__global__
void AssignFurnitures(){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    sObjs[index] = room->deviceObjs[threadIdx.x];
	__syncthreads();
}

void generate_suggestions(){
	if(room->objctNum == 0)
		return;
    int * pickedIdxs; //should be in global mem


    cudaMallocManaged(&pickedIdxs, nBlocks * nTimes * sizeof(int));
    //block1.....block2....
    for(int i=0; i<nBlocks*nTimes; i++)
        pickedIdxs[i] = rand()%room->objctNum;

    //dynamic shared mem, <<<nb, nt, sm>>>
	//obj +  temparing + room + weight
	// int sharedMem = nBlocks * (sizeof(room->objects)+ 2*sizeof(float)+ sizeof(*room) + sizeof(*deviceWeights));
	int objMem = nBlocks * room->objctNum * sizeof(singleObj);
	int floatMem = (1+nThreads) * nBlocks * sizeof(float);
    cout<<"assign begin"<<endl;
	AssignFurnitures<<<nBlocks, room->objctNum, objMem>>>();
	cudaDeviceSynchronize();
    cout<<"assign done"<<endl;
	Do_Metropolis_Hastings<<<nBlocks, nThreads, floatMem>>>(pickedIdxs, time(NULL));
	cudaDeviceSynchronize();

    room->freeMem();
	cudaFree(pickedIdxs);
	cudaFree(room);


    // for(int i=0;i<nBlocks;i++){
    //     // for(int j=0; j<numofObjs; j++)
    //         cout<<rArray[i]<<" ";
    //     cout<<endl;
    // }

}
__device__ __host__
void random_along_wall(int furnitureID) {
}


void initial_assignment(){
    for (int i = 0; i < room->freeObjNum; i++) {
    	singleObj* obj = &room->deviceObjs[room->freeObjIds[i]];
    	if (obj->adjoinWall)
    		random_along_wall(room->freeObjIds[i]);
    	else if (obj->alignedTheWall)
    		room->set_obj_zrotation(&room->deviceObjs[room->freeObjIds[i]], room->deviceWalls[rand() % room->wallNum].zrotation);
    }
    room->update_furniture_mask();
}


void parser_inputfile(const char* filename, Room * parser_inputfile) {
	ifstream instream(filename);
	string str;
	vector<vector<float>> parameters;
	vector<char> cateType;
	char  delims[] = " :,\t\n";
	char* context = nullptr;
	while (instream && getline(instream, str)) {
		if (!str.length())
			continue;
		char * charline = new char[300];
		int r = strcpy_s(charline, 300, str.c_str());
		char * itemCate = strtok_s(charline,delims,&context);
		vector<float>param;
		char * token = strtok_s(nullptr, delims, &context);
		while (token != nullptr) {
			param.push_back(atof(token));
			token = strtok_s(nullptr, delims, &context);
		}
		parameters.push_back(param);
		cateType.push_back(itemCate[0]);
	}
	instream.close();
	int itemNum = cateType.size();
	vector<vector<float>> fixedObjParams;
	vector<vector<float>> mergedObjParams;
	vector<int> groupedIds;
	int startId = 0;
	if (cateType[0] == 'r') {
		parser_inputfile->initialize_room(parameters[0][0], parameters[0][1]);
		startId = 1;
	}
	else if(!parser_inputfile->initialized)
		parser_inputfile->initialize_room();
	for (int i = startId; i < itemNum; i++) {
		switch (cateType[i])
		{
		case '#':
			break;
		//add a new wall
		case 'w':
			parser_inputfile->add_a_wall(parameters[i]);
			break;
		case 'f':
			parser_inputfile->add_an_object(parameters[i]);
			break;
		case 'p':
			parser_inputfile->add_a_focal_point(parameters[i]);
			break;
		case 'v':
            for(int k=0;k<parameters[i].size(); k++)
                weights[k] = parameters[i][k];
			break;
        default:
            break;
        }
    }
}
int main(int argc, char** argv){
    char* filename;
    /*if (argc < 2) {
        filename = new char[9];
        strcpy(filename, "input.txt");
    }
    else
        filename = argv[1];*/
	char* existance_file;
	filename = new char[100];
	existance_file = new char[100];
	int r = strcpy_s(filename, 100, "E:/layoutParam.txt");
	r = strcpy_s(existance_file, 100, "E:/fixedObj.txt");
	Room* parserRoom = new Room();
	parser_inputfile(filename, parserRoom);
	// parser_inputfile(existance_file, room, weights);
	if (parserRoom != nullptr && (parserRoom->objctNum != 0 || parserRoom->wallNum != 0))
        startToProcess(parserRoom);
	return 0;
}
