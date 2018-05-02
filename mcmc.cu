#include <iostream>
#include <string>
#include<fstream>
#include <limits.h>
#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#include "mcmc.cuh"

#define RES_NUM 1

using namespace std;
using namespace cv;

const unsigned int nBlocks = 10 ;
const unsigned int WHICH_GPU = 0;
const unsigned int nTimes =20;

void roomInitialization(Room* m_room);
void generate_suggestions(automatedLayout * layout);
extern __shared__ singleObj sObjs[];
extern __shared__ float sFloats[];
__device__ __managed__ Room* room;


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
    // room->set_obj_zrotation(&room->deviceObjs[0], PI);
    // room->set_obj_translation(&room->deviceObjs[0], -50, 0);
    // room->get_nearest_wall_dist(&room->deviceObjs[0]);
}
void startToProcess(Room * m_room, vector<float> weights){
	//cout<<"hello from mcmc"<<endl;
	setUpDevices();
	roomInitialization(m_room);
	automatedLayout * layout = new automatedLayout(weights);
	// layout->initial_assignment(m_room);
    debugCaller();
	generate_suggestions(layout);
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


void generate_suggestions(automatedLayout * layout){
	if(room->objctNum == 0)
		return;
    int * pickedIdxs; //should be in global mem


    cudaMallocManaged(&pickedIdxs, nBlocks * nTimes * sizeof(int));
    for(int i=0; i<nBlocks*nTimes; i++)
        pickedIdxs[i] = rand()%room->objctNum;

    //dynamic shared mem, <<<nb, nt, sm>>>
	//obj +  temparing + room + weight
	// int sharedMem = nBlocks * (sizeof(room->objects)+ 2*sizeof(float)+ sizeof(*room) + sizeof(*deviceWeights));
	int objMem = nBlocks * room->objctNum * sizeof(singleObj);
	int temMem = nBlocks * sizeof(float);

	AssignFurnitures<<<nBlocks, room->objctNum, objMem>>>();
	cudaDeviceSynchronize();
	Do_Metropolis_Hastings<<<nBlocks, room->objctNum, temMem>>>(layout, pickedIdxs, time(NULL));
	cudaDeviceSynchronize();

	//cudaFree(resTransAndRot);
	cudaFree(pickedIdxs);
//	cudaFree(weights);
	cudaFree(room);


    // for(int i=0;i<nBlocks;i++){
    //     // for(int j=0; j<numofObjs; j++)
    //         cout<<rArray[i]<<" ";
    //     cout<<endl;
    // }

}
__device__ __host__
void automatedLayout::random_along_wall(int furnitureID) {
}

__device__
float automatedLayout::cost_function(){
	return 0;
}
void automatedLayout::initial_assignment(){
    for (int i = 0; i < room->freeObjNum; i++) {
    	singleObj* obj = &room->deviceObjs[room->freeObjIds[i]];
    	if (obj->adjoinWall)
    		random_along_wall(room->freeObjIds[i]);
    	else if (obj->alignedTheWall)
    		room->set_obj_zrotation(&room->deviceObjs[room->freeObjIds[i]], room->deviceWalls[rand() % room->wallNum].zrotation);
    }
    room->update_furniture_mask();
}


void parser_inputfile(const char* filename, Room * room, vector<float>& weights) {
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
		room->initialize_room(parameters[0][0], parameters[0][1]);
		startId = 1;
	}
	else if(!room->initialized)
		room->initialize_room();
	for (int i = startId; i < itemNum; i++) {
		switch (cateType[i])
		{
		case '#':
			break;
		//add a new wall
		case 'w':
			room->add_a_wall(parameters[i]);
			break;
		case 'f':
			room->add_an_object(parameters[i]);
			break;
		case 'p':
			room->add_a_focal_point(parameters[i]);
			break;
		case 'v':
			weights = parameters[i];
			break;
        default:
            break;
        }
    }
    if (weights.size() < 11) {
		for (int i = weights.size(); i < 11; i++)
			weights.push_back(1.0f);
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
	Room* room = new Room();
	vector<float>weights;
	parser_inputfile(filename, room, weights);
	// parser_inputfile(existance_file, room, weights);
	room->initialize_room();
	if (room != nullptr && (room->objctNum != 0 || room->wallNum != 0))
        startToProcess(room, weights);
	// system("pause");
	return 0;
}
