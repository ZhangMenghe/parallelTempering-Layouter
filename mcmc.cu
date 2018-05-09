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

const unsigned int nBlocks = 1;
const unsigned int nThreads = 64;
const unsigned int WHICH_GPU = 0;
const unsigned int nTimes = 1;

struct sharedWrapper;
extern __shared__ sharedWrapper sWrapper[];

__device__ __managed__ float weights[11]={1.0f};
__device__ __managed__ float resTransAndRot[RES_NUM * 4];
__device__ void random_along_wall(sharedRoom * room, singleObj * obj);
__device__ void get_sum_furnitureMsk(unsigned char* mask, int colCount, int rowCount, float * res, int absThreadIdx, int threadStride);
__device__ void set_obj_zrotation(singleObj * obj, float nrot);
__device__ bool set_obj_translation(sharedRoom * room, singleObj* obj, float cx, float cy);
__device__ void storeOrigin(singleObj * obj);
struct sharedWrapper{
    sharedRoom *wRoom;//1
    singleObj *wObjs;//nblocks
    unsigned char *wMask;//nblocks
    float *wFloats;//1
    int *wPairRelation;//1
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
void rot_around_a_point(float center[3], float * x, float * y, float s, float c) {
	// translate point back to origin:
	*x -= center[0];
	*y -= center[1];

	// rotate point
	float xnew = *x * c - *y * s;
	float ynew = *x * s + *y * c;

	// translate point back:
	*x = xnew + center[0];
	*y = ynew + center[1];
}

__device__
void sumUp_dataInShare(float * data, float* res, int bound = nThreads){
    int i = (*res ==0)?0:1;
    for(; i<bound; i++)
        *res += data[i];
}
__device__
float density_function(float beta, float cost) {
    // printf("%f-%f\n", beta, cost);
	return exp2f(-beta * cost);
}

__device__
void get_random_state(curandState_t * state, int index){
    //seed, sequence number(multiple cores), offset
    curand_init((unsigned long long )clock() + index, 0, 0, state);
}
__device__
float get_normal_random(float mean, float devision,int index = blockIdx.x * blockDim.x + threadIdx.x){
    curandState_t state;
    get_random_state(&state, index);
    return curand_normal(&state) * devision/2 + mean;
}
__device__
int get_int_random(int maxLimit,int index = blockIdx.x * blockDim.x + threadIdx.x){
    curandState_t state;
    get_random_state(&state, index);
    return curand(&state)%maxLimit;
}
__device__
float get_float_random(int maxLimit, int index = blockIdx.x * blockDim.x + threadIdx.x) {
    curandState_t state;
    get_random_state(&state, index);
    return curand_uniform(&state) * maxLimit;
}

__device__
bool point_in_rectangle(float* tmpSlot, const float *vertices, float x, float y){
    int nCross = 0;
    	tmpSlot[2] = vertices[0]; tmpSlot[3] = vertices[1];
    	for (int i = 1; i <= 4; i++) {
    		tmpSlot[0] = tmpSlot[2]; tmpSlot[1] = tmpSlot[3];
    		if (i == 4) {
    			tmpSlot[2] = vertices[0]; tmpSlot[3] = vertices[1];
    		}
    		else {
    			tmpSlot[2] = vertices[2 * i]; tmpSlot[3] = vertices[2 * i + 1];
    		}
    		if (tmpSlot[1] == tmpSlot[3])
    			continue;
    		if (y < tmpSlot[1] && y < tmpSlot[3])
    			continue;
    		if (y > tmpSlot[1] && y > tmpSlot[3])
    			continue;
    		float tx = (y - tmpSlot[1])*(tmpSlot[2] - tmpSlot[0]) / (tmpSlot[3] - tmpSlot[1]) + tmpSlot[0];
    		if (tx >= x)
    			nCross++;
    	}
    	return (nCross % 2 == 1);
}
__device__
int binary_search_Inside_Point(int xl, int xr, int mid, int y, float* tmpSlot, const float* vertices){
    while (xl < xr) {
    mid = (xl + xr) / 2;
    if (point_in_rectangle(tmpSlot, vertices, mid, y))
        xl = mid + 1;
    else
        xr = mid - 1;
    }
    return xl;
}
__device__
void update_mask_by_object(unsigned char* mask, float* tmpSlot, float * vertices,
                            mRect2f boundingBox, int halfRowNum, int colNum, int absThreadIdx, int threadStride, int addition=1){

    int boundX = boundingBox.x + boundingBox.width;
    //Ideally, each thread process a row
    for(int y = boundingBox.y - absThreadIdx; y > boundingBox.y - boundingBox.height; y -= threadStride){
        //int test = 0;
        for(int x = boundingBox.x; x<boundX; x++){
            if(!point_in_rectangle(tmpSlot, vertices, x, y));
            else{
                int endIndx = binary_search_Inside_Point(x, boundX - 1, 0, y, tmpSlot, vertices);
                while(x <= endIndx){
                    mask[(halfRowNum - y) *colNum  + x] += addition;
                    x++;
                    //test += mask[(halfRowNum - y) *colNum  + x];
                }
                break;
                //while(x<boundX) {mask[/*对应位置*/] = 0; x++;}
            }
        }
        //printf("%d  - %f\n", y, test);
    }
}

__device__
void changeTemparature(float * temparature){
    int t1 = get_int_random(nBlocks);
    int t2 = t1;
    while(t2 == t1)
        t2 = get_int_random(nBlocks);
    float tmp = temparature[t1];
    temparature[t1] = temparature[t2];
    temparature[t2] = tmp;
}
__device__
int randomly_perturb(sharedRoom* room, singleObj * objs, int pickedIdx, unsigned char * mask, float* tmpSlot, float * shareState){
    int secondChangeId = -1;
    singleObj * obj = &objs[pickedIdx];
    storeOrigin(obj);

    int index = blockIdx.x * nThreads + threadIdx.x;
    // REAL RANDOM HERE
    if(threadIdx.x == 0){
        if (obj->adjoinWall)
            random_along_wall(room, obj);
        else{
            int randomMethod = (room->objctNum < 2)? 2: 3;
            switch (get_int_random(randomMethod, index)){
                // randomly rotate
                case 0:
                    if (obj->alignedTheWall)
                        set_obj_zrotation(obj, room->deviceWalls[get_int_random(room->wallNum, index)].zrotation);
                    else
                        set_obj_zrotation(obj, get_float_random(PI, index));
                    break;
                case 1:
                    while(set_obj_translation(room, obj,
                                            get_float_random(room->half_width, index),
                                            get_float_random(room->half_height, index)));
                    break;
                case 2:
                    singleObj * obj2;
                    while(1){
                        obj2 = &objs[get_int_random(room->objctNum, index)];
                        if(obj2->id == pickedIdx || obj2->adjoinWall || obj2->alignedTheWall)
                            continue;
                        storeOrigin(obj2);
                        if(!set_obj_translation(room, obj, obj2->translation[0], obj2->translation[1]))
                            continue;
                        if(!set_obj_translation(room, obj2, obj->translation[0], obj->translation[1]))
                            continue;
                        set_obj_zrotation(obj, obj2->zrotation);
                        set_obj_zrotation(obj2, obj->zrotation);
                        secondChangeId = obj2->id;
                        break;
                    }
                }//end switch
        }// end not adjoint wall
    }//end thread == 0

    else
        update_mask_by_object(mask, tmpSlot, obj->lastVertices, obj->lastBoundingBox,
                              room->rowCount/2, room->colCount,
                              threadIdx.x - 1, nThreads-1, -1);


    __syncthreads();

    update_mask_by_object(mask, tmpSlot, obj->vertices, obj->boundingBox,
                          room->rowCount/2, room->colCount,
                          threadIdx.x, nThreads, 1);


    tmpSlot[threadIdx.x] = 0;
    get_sum_furnitureMsk(mask, room->colCount, room->rowCount, &tmpSlot[threadIdx.x], threadIdx.x, nThreads);

    __syncthreads();

    sumUp_dataInShare(tmpSlot, &shareState[blockIdx.x]);
    // printf("%f\n", tmpSlot[0]);
    return secondChangeId;

}

__device__
void random_along_wall(sharedRoom * room, singleObj * obj){

}
__device__
bool set_obj_translation(sharedRoom * room, singleObj* obj, float cx, float cy){
    cx = (get_int_random(2) == 0)?cx:-cx;
    cy = (get_int_random(2) == 0)?cy:-cy;
    float halfw = obj->boundingBox.width/2, halfh = obj->boundingBox.height/2;
    if( cx + halfw > room->half_width || cx-halfw < -room->half_width
     || cy + halfh > room->half_height || cy-halfh < -room->half_height)
     return false;

    float movex = cx - obj->translation[0], movey = cy-obj->translation[1];
    obj->translation[0] = cx; obj->translation[1]=cy;
    for(int i=0; i<4; i++){
        obj->vertices[2*i]+=movex;
        obj->vertices[2*i + 1] += movey;
    }
    obj->boundingBox.x += movex; obj->boundingBox.y += movey;
    return true;
}
__device__
void set_obj_zrotation(singleObj * obj, float nrot) {
	float oldRot = obj->zrotation;
	nrot = remainderf(nrot, 2*PI);
	obj->zrotation = nrot;
	float gap = obj->zrotation - oldRot;
	float s = sinf(gap); float c=cosf(gap);
	float minx = INFINITY, maxx =-INFINITY, miny=INFINITY, maxy = -INFINITY;
	for(int i=0; i<4; i++){
		rot_around_a_point(obj->translation, &obj->vertices[2*i], &obj->vertices[2*i+1], s, c);
		minx = (obj->vertices[2*i] < minx)? obj->vertices[2*i]:minx;
		maxx = (obj->vertices[2*i] > maxx)? obj->vertices[2*i]:maxx;
		miny = (obj->vertices[2*i + 1] < miny)? obj->vertices[2*i+1]:miny;
		maxy = (obj->vertices[2*i + 1] > maxy)? obj->vertices[2*i+1]:maxy;
	}
	obj->boundingBox.x = minx; obj->boundingBox.y=maxy;
	obj->boundingBox.width = maxx-minx; obj->boundingBox.height = maxy-miny;
}

__device__
void initial_assignment(sharedRoom* room, singleObj * objs,  unsigned char * mask, float* tmpSlot){
    if(threadIdx.x < room->objctNum){
        singleObj * obj = &objs[threadIdx.x];
        if (obj->adjoinWall)
            random_along_wall(room, obj);
        else if (obj->alignedTheWall)
            set_obj_zrotation(obj, room->deviceWalls[get_int_random(room->wallNum)].zrotation);
    }

    __syncthreads();
    //all threads to do update masks
    for (int i = 0; i < room->freeObjNum; i++) {
        update_mask_by_object(mask, tmpSlot, objs[room->freeObjIds[i]].vertices, objs[room->freeObjIds[i]].boundingBox,
                              room->rowCount/2, room->colCount,
                              threadIdx.x, nThreads, 1);
    }
    tmpSlot[threadIdx.x] = 0;

    //get_sum_furnitureMsk(mask, room->colCount, room->rowCount, &tmpSlot[threadIdx.x], threadIdx.x, threadStride);
    __syncthreads();
}

__device__
void storeOrigin(singleObj * obj){
    for(int i=0;i<8;i++)
        obj->lastVertices[i] = obj->vertices[i];
    for(int i=0;i<3;i++)
        obj->lastTransAndRot[i] = obj->translation[i];
    obj->lastTransAndRot[3] = obj->zrotation;
    obj->lastBoundingBox = obj->boundingBox;
}

__device__
void restoreOrigin(singleObj * obj){
    for(int i=0;i<8;i++)
        obj->vertices[i] = obj->lastVertices[i];
    for(int i=0;i<3;i++)
        obj->translation[i] = obj->lastTransAndRot[i];
    obj->zrotation = obj->lastTransAndRot[3];
    obj->boundingBox = obj->lastBoundingBox;
    //TODO:should restore mask as well
}
__device__
void getTemporalTransAndRot(){

}
__device__
float t(float d, float m, float M, int a = 2){
    if (d < m)
		return powf((d / m), float(a));
	else if (d > M)
		return powf((M / d), float(a));
	else
		return 1.0f;
}

__device__
float dist_between_points(const float* pos1, const float* pos2) {
	return sqrtf(powf((pos1[0] - pos2[0]),2.0f) + powf((pos1[1] - pos2[1]),2.0f) +powf((pos1[2] - pos2[2]),2.0f));
}


__device__
void get_sum_furnitureMsk(unsigned char* mask, int colCount, int rowCount, float * res, int absThreadIdx, int threadStride){
    for(int row = absThreadIdx; row<rowCount; row+=threadStride){
        for(int col =0; col<colCount; col++){
            if(mask[row*colCount + col] > 0)
                *res+=1;
        }
    }
    //printf("%d - %f\n", threadIdx.x, *res);
}
__device__
float get_nearest_wall_dist(singleObj * obj, wall* deviceWalls, int wallNum) {
	float min_dist = INFINITY, dist;
	for (int i = 0; i < wallNum; i++) {
		dist = fabsf(deviceWalls[i].a * obj->translation[0] + deviceWalls[i].b * obj->translation[1] + deviceWalls[i].c) / sqrtf(deviceWalls[i].a * deviceWalls[i].a + deviceWalls[i].b * deviceWalls[i].b);
		if (dist < min_dist) {
			min_dist = dist;
			obj->nearestWall = i;
		}
	}
	// printf("%d : %f\n", obj->nearestWall, min_dist);
	return min_dist;
}

__device__
void get_obj_reflection(singleObj * obj, const float* focal){
    if(focal[0] == INFINITY)
        return;
    if(REFLECT_K == 0){
        obj->refRot = PI - obj->zrotation;
        obj->refPos[0] = obj->translation[0];
        obj->refPos[1] = 2*focal[1] -  obj->translation[1];
    }
    else if(REFLECT_K== INFINITY){
            obj->refRot = - obj->zrotation;
            obj->refPos[0] = 2*focal[0] -  obj->translation[0];
            obj->refPos[1] = obj->translation[1];
        }
    else{
        float invk = 1/ (REFLECT_K +0.00001f) ;
        float b = focal[1] - focal[0] * REFLECT_K;
        float x = 2 * obj->translation[1] + (invk - REFLECT_K)*obj->translation[0] - 2 * b;
        float y = -invk * x + obj->translation[1] + invk*obj->translation[0];
        obj->refPos[0] = x;
        obj->refPos[1]  = y;
        obj->refRot = PI - obj->zrotation - 2*atan2f(obj->translation[1]-y, obj->translation[0]-x);
    }
}

//Clearance :
//Mcv(I) that minimize the overlap between furniture(with space)
__device__
void cal_clearance_violation(float overlappingArea, float& mcv){
    mcv = sWrapper[0].wRoom->indepenFurArea - overlappingArea;
    mcv = (mcv < 0)? 0 : mcv;
}
//Circulation:
//Mci support circulation through the room and access to all of the furniture.
__device__
void cal_circulation_term(unsigned char* mask, unsigned char* dest, float& mci){
    mci = 0;
}
//Pairwise relationships:
//Mpd: for example  coffee table and seat
//mpa: relative direction constraints
__device__
void cal_pairwise_relationship(singleObj* objs, float& mpd, float& mpa){
    singleObj * obj1, *obj2;
    float cosfg2;
    for(int i=0; i<sWrapper[0].wRoom->pairNum; i++){
        obj1 = &objs[sWrapper[0].wPairRelation[4*i]];
        obj2 = &objs[sWrapper[0].wPairRelation[4*i+1]];
        //printf("%d - %d - %d - %d\n",blockIdx.x, threadIdx.x, obj1->id, obj2->id);
        mpd -= t(dist_between_points(obj1->translation, obj2->translation),
                sWrapper[0].wPairRelation[4*i + 2],
                sWrapper[0].wPairRelation[4*i + 3]);
        cosfg2 = powf((sinf(obj1->zrotation) * sinf(obj2->zrotation)
                    + cosf(obj1->zrotation) * cosf(obj2->zrotation)),2.0f);
        mpa -= 8 * powf(cosfg2, 2) - 8 * cosfg2;
    }
}
//Conversation
//Mcd:group a collection of furniture items into a conversation area
__device__
void cal_conversation_term(singleObj *objs, const int*ids, int memNum, float& mcd, float& mca){
    singleObj * obj, *obj2;
    float cosfg, cosgf;

    for(int i=0;i<memNum-1;i++){
        obj = &objs[ids[i]];
        if(obj->catalogId == TYPE_CHAIR){
            for(int j=i+1; j<memNum; j++){
                obj2 = &objs[ids[j]];
                if(obj->catalogId == TYPE_CHAIR){
                    mcd += t(dist_between_points(obj->translation, obj2->translation), CONVERSATION_M_MIN, CONVERSATION_M_MAX);
                    cosfg = cosf(obj->zrotation) *(obj2->translation[0] - obj->translation[0])
                                + sinf(obj->zrotation) * (obj2->translation[1] - obj->translation[1]);
                    cosfg = cosf(obj2->zrotation) *(obj->translation[0] - obj2->translation[0])
                                + sinf(obj2->zrotation) * (obj->translation[1] - obj2->translation[1]);
                    mca -= (cosfg + 1) *(cosgf +1);
                }
            }
        }

    }
}
//balance:
//place the mean of the distribution of visual weight at the center of the composition
__device__
void cal_balance_term(const singleObj * obj, float &mvb){
    float centroid[3] = {.0f};
    centroid[0] += obj->area * obj->translation[0];
    centroid[1] += obj->area * obj->translation[1];
    centroid[2] += obj->area * obj->translation[2];
    centroid[0] /= sWrapper[0].wRoom->indepenFurArea;centroid[1] /= sWrapper[0].wRoom->indepenFurArea;centroid[2] /= sWrapper[0].wRoom->indepenFurArea;
    //printf("%f - %f - %f- %f\n", centroid[0],centroid[1],centroid[2],sWrapper[0].wRoom->indepenFurArea);
    mvb = dist_between_points(centroid, sWrapper[0].wRoom->RoomCenter);
}
//Alignment:
//compute furniture alignment term
__device__
void cal_alignment_term(singleObj * objs, int gid, int mid, float& mfa, float&mwa){
    singleObj *obj1 = &objs[threadIdx.x] , *obj2;

    get_nearest_wall_dist(obj1, sWrapper[0].wRoom->deviceWalls, sWrapper[0].wRoom->wallNum);
    if(!obj1->adjoinWall)
        mwa -= cosf(4 * (obj1->zrotation
                - sWrapper[0].wRoom->deviceWalls[obj1->nearestWall].zrotation
                - PI/2));

    for(int k = mid; k<sWrapper[0].wRoom->groupMap[gid].memNum-1; k++){
        obj2 = &objs[threadIdx.x + k];
        mfa -= cosf(4 * (obj1->zrotation - obj2->zrotation));
    }
}
//Emphasis:
//compute focal center
__device__
void cal_emphasis_term(singleObj * obj, const float * focal, float& mef){
    if(focal[0] == INFINITY)
        return;
    get_obj_reflection(obj, focal);
    float dist = dist_between_points(focal, obj->translation);
    mef -= (focal[0] - obj->translation[0])/dist * cosf(obj->zrotation)
          + (focal[1] - obj->translation[1])/dist * sinf(obj->zrotation);
}
__device__
void cal_emphasis_term2(singleObj *objs, groupMapStruct* gmap, float& msy, float gamma = 1){
    singleObj *obj, *obj2;
    float maxS = -INFINITY, tmpS;
    for(int i=0; i<gmap->memNum-1; i++){
        obj = &objs[gmap->objIds[i]];
        for(int j=i+1; j<gmap->memNum; j++){
            obj2 = &objs[gmap->objIds[j]];
            tmpS = cosf(obj->zrotation - obj2->refRot) - gamma*dist_between_points(gmap->focal, obj2->refPos);
            maxS = (tmpS > maxS)? tmpS:maxS;
        }
        msy -= maxS;
        maxS = -INFINITY;
    }
}
__device__
void calCostParam(singleObj * objs, float* costList){
    singleObj * obj = &objs[threadIdx.x];
    for(int i=0; i<sWrapper[0].wRoom->groupNum; i++){
        for(int j=0; j<sWrapper[0].wRoom->groupMap[i].memNum; j++){
            if(sWrapper[0].wRoom->groupMap[i].objIds[j] == threadIdx.x){
                cal_emphasis_term(obj, sWrapper[0].wRoom->groupMap[i].focal, costList[1]);
                cal_alignment_term(objs, i, j, costList[3], costList[4]);
                cal_balance_term(obj, costList[7]);
            }
        }
    }
    //Each thread work on a group
    if(threadIdx.x < sWrapper[0].wRoom->groupNum){
        cal_conversation_term(objs, sWrapper[0].wRoom->groupMap[threadIdx.x].objIds,
                                sWrapper[0].wRoom->groupMap[threadIdx.x].memNum,
                              costList[5], costList[6]);
        cal_emphasis_term2(objs, &sWrapper[0].wRoom->groupMap[threadIdx.x], costList[2]);
    }
}
__device__
float getWeightedCost(singleObj* objs, float* costList, float * shareState, int objNum){
    costList[threadIdx.x] = 0;
    if(threadIdx.x < objNum)
        calCostParam(objs, costList);
    else if(threadIdx.x == objNum)
        cal_pairwise_relationship(objs, costList[10], costList[11]);
    else {
        //cal_circulation_term(shareState[0], costList[8]);
        //shareState[0] store the sum of furniture mask
        cal_clearance_violation(shareState[blockIdx.x], costList[9]);
    }
    //else do nothing, empty the first #numofObjs slots
    // TODO: PLEASE MAKE SURE sumUp_dataInShare
    __syncthreads();
    if(threadIdx.x < WEIGHT_NUM)
        shareState[blockIdx.x] += weights[threadIdx.x] * costList[1+threadIdx.x];
    __syncthreads();
    return shareState[blockIdx.x];
}

__device__
void Metropolis_Hastings(float* costList,float* shareState, float* temparature, int* pickedIdxs, int objIndexId){
    float cpost, p0, p1, alpha;
    int startId = blockIdx.x * nThreads;
    int index = startId + threadIdx.x;
    int secondChangeId;
    //sharedRoom* room, singleObj * objs,  unsigned char * mask, float* tmpSlot, int threadStride)
    initial_assignment(sWrapper[0].wRoom, &sWrapper[0].wObjs[objIndexId],
                        &sWrapper[0].wMask[sWrapper[0].wRoom->mskCount * blockIdx.x],
                        &costList[startId]);

    sumUp_dataInShare(&costList[startId], &shareState[blockIdx.x]);
    costList[index] = 0;
    float cpre = getWeightedCost(&sWrapper[0].wObjs[objIndexId], &costList[startId], shareState, sWrapper[0].wRoom->objctNum);
    //first thread cost is the best cost of block
    costList[startId] = cpre;
    for(int nt = 0; nt<nTimes; nt++){
        if(pickedIdxs[blockIdx.x] == threadIdx.x){
            if(nBlocks>1 && nt % 10 == 0)
                changeTemparature(temparature);
            p0 = density_function(temparature[blockIdx.x], cpre);
        }
        shareState[blockIdx.x] = 0;
        secondChangeId = randomly_perturb(sWrapper[0].wRoom, &sWrapper[0].wObjs[objIndexId],pickedIdxs[blockIdx.x],
                        &sWrapper[0].wMask[sWrapper[0].wRoom->mskCount * blockIdx.x], &costList[startId], shareState);

        cpost = getWeightedCost(&sWrapper[0].wObjs[objIndexId], &costList[startId], shareState, sWrapper[0].wRoom->objctNum);
        costList[index] = 0;
        if(pickedIdxs[blockIdx.x] == threadIdx.x){
            p1 = density_function(temparature[blockIdx.x], cpost);
            alpha = fminf(1.0f, p1/p0);
            if(alpha > THREADHOLD_T){
                restoreOrigin(&sWrapper[0].wObjs[objIndexId + pickedIdxs[blockIdx.x]]);
                if(secondChangeId!=-1)
                    restoreOrigin(&sWrapper[0].wObjs[objIndexId + secondChangeId]);
            }
            else if(cpost < costList[blockIdx.x]){
                getTemporalTransAndRot();
                costList[startId] = cpost;
                cpre = cpost;
            }
        }
        else{
            pickedIdxs[blockIdx.x] = get_int_random(sWrapper[0].wRoom->objctNum, index);
            shareState[blockIdx.x] = 0;
        }
        __syncthreads();
    }
}

__global__
void Do_Metropolis_Hastings(sharedWrapper *gWrapper){
    sWrapper[0] = *gWrapper;
    if(blockIdx.x !=0 ){
        if(threadIdx.x < sWrapper[0].wRoom->objctNum){
            int objId = blockIdx.x * sWrapper[0].wRoom->objctNum + threadIdx.x;
            sWrapper[0].wObjs[objId] = sWrapper[0].wObjs[threadIdx.x];
        }
        //can be optimized
        if(threadIdx.x == 0){
            for(int i=0; i<sWrapper[0].wRoom->mskCount; i++)
                sWrapper[0].wMask[blockIdx.x *sWrapper[0].wRoom->mskCount + i] = sWrapper[0].wMask[i];
        }
    }

    float* costList = sWrapper[0].wFloats;
    float * shareState = (float *)& costList[nBlocks * nThreads];
    float* temparature = (float *) & shareState[nBlocks];
    int* pickedIdxs = (int *)& temparature[nBlocks];
    shareState[blockIdx.x] = 0;
	temparature[blockIdx.x] = -get_float_random(10);
    pickedIdxs[blockIdx.x] = get_int_random(sWrapper[0].wRoom->objctNum);
    // printf("%d\n", pickedIdxs[blockIdx.x]);
    Metropolis_Hastings(costList,shareState, temparature, pickedIdxs, blockIdx.x * sWrapper[0].wRoom->objctNum);
    __syncthreads();
}



void generate_suggestions(Room * m_room){
    sharedWrapper *gWrapper;
    cudaMallocManaged(&gWrapper,  sizeof(sharedWrapper));

    cudaMallocManaged(&gWrapper->wRoom, sizeof(sharedRoom));
    m_room->CopyToSharedRoom(gWrapper->wRoom);

    int objMem = nBlocks *m_room->objctNum * sizeof(singleObj);
    cudaMallocManaged(&gWrapper->wObjs, objMem);
	for(int i=0; i<m_room->objctNum; i++)
		gWrapper->wObjs[i] = m_room->objects[i];

    int tMem = m_room->colCount * m_room->rowCount * sizeof(unsigned char);
    cudaMallocManaged(&gWrapper->wMask, nBlocks *tMem);
    cudaMemcpy(gWrapper->wMask, m_room->furnitureMask, tMem, cudaMemcpyHostToDevice);

	int floatMem =  nBlocks *(3+nThreads) * sizeof(float);
    cudaMallocManaged(&gWrapper->wFloats, floatMem);

    int pairMem = m_room->actualPairs.size() * 4 * sizeof(int);
    cudaMallocManaged(&gWrapper->wPairRelation, pairMem);
    for(int i=0;i<m_room->actualPairs.size();i++){
        for(int j=0;j<4;j++)
            gWrapper->wPairRelation[4*i+j]= m_room->actualPairs[i][j];
    }


	Do_Metropolis_Hastings<<<nBlocks, nThreads, sizeof(*gWrapper)>>>(gWrapper);
	cudaDeviceSynchronize();

    cudaFree(gWrapper->wRoom);
    cudaFree(gWrapper->wObjs);
    cudaFree(gWrapper->wMask);
    cudaFree(gWrapper->wFloats);
    cudaFree(gWrapper);
}

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
void setupDebugRoom(Room* room){
    float wallParam1[] = {-200, 150, 200, 150};
    float wallParam2[] = {-200, -150, 200, -150};
    float objParam[] = {0, 0, 100, 50, 0, 0, 10};
    float fpParam[] = {0, 150, 0};
    float mWeights[] = {1, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 0.5};

    room->initialize_room();
    room->add_a_wall(vector<float>(wallParam1,wallParam1 + 4));
    room->add_a_wall(vector<float>(wallParam2,wallParam2 + 4));
    room->add_an_object(vector<float>(objParam,objParam + 7));
    room->add_an_object(vector<float>(objParam,objParam + 7));
    room->add_a_focal_point(vector<float>(fpParam,fpParam + 3));
    //room->objects[0].alignedTheWall = true;
    for(int i=0;i<11;i++)
        weights[i] = mWeights[i];
    for(int i=0; i< room->objctNum-1; i++){
        for(int j=i+1; j<room->objctNum; j++)
            room->set_objs_pairwise_relation(room->objects[i], room->objects[j]);
    }

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
    setupDebugRoom(parserRoom);
	// parser_inputfile(filename, parserRoom);
	// parser_inputfile(existance_file, room, weights);
	// if (parserRoom != nullptr && (parserRoom->objctNum != 0 || parserRoom->wallNum != 0))
    startToProcess(parserRoom);
	return 0;
}
