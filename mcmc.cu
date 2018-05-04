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
#define MAX_THREAD_NUM 64
using namespace std;

const unsigned int nBlocks = 3;
const unsigned int nThreads = 16;
const unsigned int WHICH_GPU = 0;
const unsigned int nTimes = 1;
struct sharedWrapper;
extern __shared__ sharedWrapper sWrapper[];
// extern __shared__ sharedRoom sRoom[];
// extern __shared__ singleObj sObjs[];
// extern __shared__ unsigned char sMask[];
// extern __shared__ float sFloats[];

// __device__ __managed__ Room* room;
__device__ __managed__ float weights[11]={1.0f};
__device__ __managed__ float resTransAndRot[RES_NUM * 4];
struct sharedWrapper{
    sharedRoom *wRoom;
    singleObj *wObjs;
    unsigned char *wMask;
    float *wFloats;
};
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

__device__
float density_function(float beta, float cost) {
    // printf("%f-%f\n", beta, cost);
	return exp2f(-beta * cost);
}

__device__
float get_randomNum(unsigned int seed, int maxLimit) {
  curandState_t state;
  //seed, sequence number(multiple cores), offset
  curand_init(seed, 0,0, &state);
  return curand(&state) % maxLimit;
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
    if(threadIdx.x > consStartId)
        costList[threadIdx.x] = threadIdx.x;
        // room->get_constrainTerms(costList, threadIdx.x - consStartId);
    //else do nothing, empty the first #numofObjs slots
    __syncthreads();
    float res = 0;
    for(int i=0; i<WEIGHT_NUM; i++)
        res += weights[i] * costList[consStartId + i];
    return res;
}

__device__
void Metropolis_Hastings(float* costList, float* temparature, unsigned int seed){
    float cpost, p0, p1, alpha;
    int startId = blockIdx.x * nThreads;
    int index = startId + threadIdx.x;
    costList[index] = 0;
    float cpre = 0;//getWeightedCost(&costList[startId], sRoom[0].objctNum);
    //first thread cost is the best cost of block
    costList[startId] = cpre;
    for(int nt = 0; nt<nTimes; nt++){
        //if(pickedIdAddr[nt] == threadIdx.x){
            if(nt % 10 == 0)
                changeTemparature(temparature, seed+blockIdx.x);
            p0 = density_function(temparature[blockIdx.x], cpre);
            randomly_perturb(/*original keep sth to restore*/);
        //}
        __syncthreads();
        cpost = getWeightedCost(costList, startId);
        costList[index] = 0;
        //per block operation
        //if(pickedIdAddr[nt] == threadIdx.x){
            p1 = density_function(temparature[blockIdx.x], cpost);
            alpha = fminf(1.0f, p1/p0);
            if(alpha > THREADHOLD_T)
                restoreOrigin();
            else if(cpost < costList[blockIdx.x]){
                getTemporalTransAndRot();
                costList[startId] = cpost;
                cpre = cpost;
            }
        //}
    }
}

__global__
void Do_Metropolis_Hastings(sharedWrapper *gWrapper, unsigned int seed){
    sWrapper[0] = *gWrapper;
    printf("wall : %d\n",sWrapper[0].wRoom->wallNum);

	float* costList = sWrapper[0].wFloats;
    float* temparature = (float *) & costList[nBlocks * nThreads];
    int* pickedIdxs = (int *)& temparature[nBlocks];
	temparature[blockIdx.x] = -get_randomNum(seed+blockIdx.x, 100) / 10;
    pickedIdxs[blockIdx.x] = int(get_randomNum(seed+blockIdx.x, sWrapper[0].wRoom->objctNum));
    printf("picked: %d\n",  sWrapper[0].wRoom->objctNum);
    // Metropolis_Hastings(costList, temparature, seed);
    __syncthreads();
}



void generate_suggestions(Room * m_room){
    sharedWrapper *gWrapper;
    cudaMallocManaged(&gWrapper,  sizeof(sharedWrapper));

    cudaMallocManaged(&gWrapper->wRoom, sizeof(sharedRoom));
    m_room->CopyToSharedRoom(gWrapper->wRoom);

    int objMem = m_room->objctNum * sizeof(singleObj);
    cudaMallocManaged(&gWrapper->wObjs,  m_room->objctNum * sizeof(singleObj));
	for(int i=0; i<m_room->objctNum; i++)
		gWrapper->wObjs[i] = m_room->objects[i];

    int tMem = m_room->colCount * m_room->rowCount * sizeof(unsigned char);
    cudaMallocManaged(&gWrapper->wMask, tMem);
    cudaMemcpy(gWrapper->wMask, m_room->furnitureMask, tMem, cudaMemcpyHostToDevice);

	int floatMem =  nBlocks *(2+nThreads) * sizeof(float);
    cudaMallocManaged(&gWrapper->wFloats, floatMem);

	Do_Metropolis_Hastings<<<nBlocks, nThreads, sizeof(*gWrapper)>>>(gWrapper, time(NULL));
	cudaDeviceSynchronize();

    cudaFree(gWrapper->wRoom);
    cudaFree(gWrapper->wObjs);
    cudaFree(gWrapper->wMask);
    cudaFree(gWrapper->wFloats);
    cudaFree(gWrapper);
}

__device__ __host__
void random_along_wall(int furnitureID) {
}


// void initial_assignment(){
//     for (int i = 0; i < room->freeObjNum; i++) {
//     	singleObj* obj = &room->deviceObjs[room->freeObjIds[i]];
//     	if (obj->adjoinWall)
//     		random_along_wall(room->freeObjIds[i]);
//     	else if (obj->alignedTheWall)
//     		room->set_obj_zrotation(&room->deviceObjs[room->freeObjIds[i]], room->deviceWalls[rand() % room->wallNum].zrotation);
//     }
//     room->update_furniture_mask();
// }

void startToProcess(Room * m_room){
    if(m_room->objctNum == 0)
        return;
	setUpDevices();

    clock_t start, finish;
    float costtime;
    start = clock();

	generate_suggestions(m_room);

    finish = clock();
    costtime = (float)(finish - start) / CLOCKS_PER_SEC;
    cout<<"Runtime: "<<costtime<<endl;
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
