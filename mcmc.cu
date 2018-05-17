#include <iostream>
#include "room.cuh"
#include "cudaroom.cuh"
#include "utils.cuh"
#include "constrainTerms.cuh"
#include "hostUtils.h"
using namespace std;

#define THREADHOLD_T 0.7

const unsigned int nBlocks = 10;
const unsigned int nThreads = 32;//it's werid
const unsigned int WHICH_GPU = 0;

struct sharedWrapper;
extern __shared__ sharedWrapper sWrapper[];

struct sharedWrapper{
    int nTimes;
    sharedRoom *wRoom;//1
    singleObj *wObjs;//nblocks
    unsigned char *wMask;//nblocks
    unsigned char* backMask;//nblocks
    float * wmaskArea;//nblocks * 1
    float *wFloats;//1
    int *wPairRelation;//1
    float * resTransAndRot;//1 for all objs and all blocks
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
void random_along_wall(sharedRoom * room, singleObj * obj){
    wall * swall = &room->deviceWalls[get_int_random(room->wallNum)];
    float mwidth, mheight;
    if(get_int_random(2)==0){
        mwidth = obj->objWidth; mheight = obj->objHeight;
        set_obj_zrotation(obj, swall->zrotation);
    }else{
        mwidth = obj->objHeight; mheight = obj->objWidth;
        set_obj_zrotation(obj, PI/2-swall->zrotation);
    }

    float width_ran = swall->width - mwidth, height_ran =swall->width-mheight;
    float rh, rw;
    int mp = (swall->translation[0] >0 || swall->translation[1]>0)? -1:1;
    if(fabsf(swall->b) < 0.01){
        rh = min(swall->vertices[1], swall->vertices[3]) + get_float_random(height_ran) + obj->boundingBox.height/2;
        set_obj_translation(room, obj, swall->translation[0] + mp*(mwidth/2+0.01), rh);
    }
    else if(fabsf(swall->a) < 0.01){
        rw = min(swall->vertices[0], swall->vertices[2]) + get_float_random(width_ran) + obj->boundingBox.width/2;
        set_obj_translation(room, obj, rw,swall->translation[1] + mp*(mheight/2+0.01) );
    }
    else{
        //TODO:
        printf("CANNOT ACCEPT OBLIQUE WALL\n");
    }
}

__device__
void initial_assignment(sharedRoom* room, singleObj * objs,
                        unsigned char * mask,unsigned char * backupMask, float* tmpSlot){
    if(threadIdx.x < room->objctNum){
        singleObj * obj = &objs[threadIdx.x];
        if (obj->adjoinWall)
            random_along_wall(room, obj);

        else if (obj->alignedTheWall)
            set_obj_zrotation(obj, room->deviceWalls[get_int_random(room->wallNum)].zrotation);

        //INITIALIZE COST
        int singleSize = room->objctNum * 4 + 1;
        for(int i=0; i<MAX_KEPT_RES; i++){
            sWrapper[0].resTransAndRot[singleSize*i + threadIdx.x * 4] = INFINITY;
        }

    }
    __syncthreads();
    //all threads to do update masks
    for (int i = 0; i < room->objctNum; i++) {
        update_mask_by_object(mask, tmpSlot, objs[i].vertices, objs[i].boundingBox,
                              room->rowCount/2, room->colCount,
                              threadIdx.x, nThreads, 1);

        mRect2f rect = get_circulate_boundingbox(room, &objs[i].boundingBox);

        update_mask_by_boundingBox(backupMask, rect, room->rowCount/2, room->colCount, threadIdx.x, nThreads, -1);
    }
    __syncthreads();

    sumUpMask(room, mask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x], nThreads);
    sumUpMask(room, backupMask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x+1], nThreads);
    // if(threadIdx.x ==0){
    //     printf("sum of mask inital: %f\n", sWrapper[0].wmaskArea[2*blockIdx.x]);
    // }
    // storeOrigin(&objs[0]);
    // change_an_obj_mask(room, &objs[0], mask, tmpSlot, threadIdx.x, nThreads);
    // __syncthreads();
    //
    // sumUpMask(room, mask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x], nThreads);
    //
    // if(threadIdx.x ==0){
    //     printf("sum of mask inital - test: %f\n", sWrapper[0].wmaskArea[2*blockIdx.x]);
    // }
}

__device__
void getTemporalTransAndRot(sharedRoom * room, singleObj* objs, float * results, float cost){
    float maxCost = results[0];
    int i = 1, maxPos = 0, singleSize = room->objctNum * 4 + 1;
    for(i=1; i<MAX_KEPT_RES; i++){
        if(maxCost == INFINITY)
            break;
        if(results[singleSize * i] >maxCost){
            maxPos=i; maxCost = results[singleSize * i];
        }
    }
    if(cost < maxCost){
        int baseId = singleSize * maxPos;
        results[baseId] = cost;
        for(int i=0; i<room->objctNum; i++){
            results[baseId + 4*i + 1] = objs[i].translation[0];
            results[baseId + 4*i + 2] = objs[i].translation[1];
            results[baseId + 4*i + 3] = objs[i].translation[2];
            results[baseId + 4*i + 4] = objs[i].zrotation;
        }
    }
}

__device__
int randomly_perturb(sharedRoom* room, singleObj * objs, int pickedIdx,
                    unsigned char * mask, unsigned char* backupMask,float* tmpSlot){
    int secondChangeId = -1;
    singleObj * obj = &objs[pickedIdx];
    storeOrigin(obj);
    int index = blockIdx.x * nThreads + threadIdx.x;
    // REAL RANDOM HERE
    if(threadIdx.x == 0){
        if (obj->adjoinWall)
            random_along_wall(room, obj);
        else{
            int randomMethod = (room->objctNum < 2 || obj->alignedTheWall)? 2: 3;
            switch (get_int_random(2, index)){
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
                                            get_float_random(room->half_height, index),true));
                    break;
                case 2:
                    singleObj * obj2;
                    int trytimes = 0;
                    // float tmpx = obj->translation[0], tmpy=obj->translation[1], tmprot = obj->zrotation;
                    while(trytimes++ < 5){
                        obj2 = &objs[get_int_random(room->objctNum, index)];
                        if(obj2->id == pickedIdx || obj2->adjoinWall || obj2->alignedTheWall)
                            continue;
                        storeOrigin(obj2);

                        if(!set_obj_translation(room, obj, obj2->translation[0], obj2->translation[1]))
                            continue;
                        if(!set_obj_translation(room, obj2, obj->lastTransAndRot[0], obj->lastTransAndRot[1])){
                            set_obj_translation(room, obj, obj->lastTransAndRot[0], obj->lastTransAndRot[1]);
                            continue;
                        }
                        break;
                    }
                    if(trytimes >= 5)
                        while(set_obj_translation(room, obj,
                                                get_float_random(room->half_width, index),
                                                get_float_random(room->half_height, index),true));
                    else{
                        set_obj_zrotation(obj, obj2->zrotation);
                        set_obj_zrotation(obj2, obj->lastTransAndRot[3]);
                        secondChangeId = obj2->id;
                    }
                    break;
                default:
                    break;
                }//end switch
        }// end not adjoint wall
    }//end thread == 0

        // change_an_obj_mask(room, obj, mask, tmpSlot, threadIdx.x, nThreads);
        // change_an_obj_backupMask(room, obj, backupMask, nThreads);
        // if(secondChangeId!=-1){
        //     change_an_obj_mask(room, &objs[secondChangeId], mask, tmpSlot, threadIdx.x, nThreads);
        //     change_an_obj_backupMask(room, &objs[secondChangeId], backupMask, nThreads);
        // }

    memset(mask, 0, room->mskCount * sizeof(unsigned char));
    // __syncthreads();
    // sWrapper[0].wmaskArea[2*blockIdx.x] = 0;
    // sumUpMask(room, mask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x], nThreads);
    // sumUpMask(room, backupMask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x+1], nThreads);
    for (int i = 0; i < room->objctNum; i++) {
        update_mask_by_object(mask, tmpSlot, objs[i].vertices, objs[i].boundingBox,
                              room->rowCount/2, room->colCount,
                              threadIdx.x, nThreads, 1);

        mRect2f rect = get_circulate_boundingbox(room, &objs[i].boundingBox);

        update_mask_by_boundingBox(backupMask, rect, room->rowCount/2, room->colCount, threadIdx.x, nThreads, -1);
    }
    __syncthreads();

    sumUpMask(room, mask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x], nThreads);
    sumUpMask(room, backupMask, tmpSlot, &sWrapper[0].wmaskArea[2*blockIdx.x+1], nThreads);
    // if(threadIdx.x ==0){
    //     printf("loc and pos: %f, %f, %f\n",obj->translation[0], obj->translation[1], obj->zrotation );
    //     printf("sum of mask: %f\n", sWrapper[0].wmaskArea[2*blockIdx.x]);
    // }
    return secondChangeId;

}

__device__
void Metropolis_Hastings(float* costList, float* temparature, int*pickedupIds){
    float cpost, p0, p1, alpha;
    sharedRoom * room = sWrapper[0].wRoom;
    singleObj * objsBlock = &sWrapper[0].wObjs[blockIdx.x * room->objctNum];
    int startId = blockIdx.x * nThreads;
    int index = startId + threadIdx.x;
    int maskStart =sWrapper[0].wRoom->mskCount * blockIdx.x;
    int secondChangeId,pickedId;

    //sharedRoom* room, singleObj * objs,  unsigned char * mask, float* tmpSlot, int threadStride)
    initial_assignment(room, objsBlock,
                        &sWrapper[0].wMask[maskStart], &sWrapper[0].backMask[maskStart], &costList[startId]);


    getWeightedCost(room, objsBlock, sWrapper[0].wPairRelation,&sWrapper[0].wmaskArea[2*blockIdx.x], &costList[startId]);
    __syncthreads();

    float cpre = sumUp_weighted_dataInShare(&costList[startId+1], weights, WEIGHT_NUM);
    getTemporalTransAndRot(room, objsBlock, sWrapper[0].resTransAndRot, cpre);
    // if(threadIdx.x == 0)
    //     displayResult(costList, weights);
    for(int nt = 0; nt<sWrapper[0].nTimes; nt++){
        if(threadIdx.x == 0){
            if(nBlocks>1 && nt % 10 == 0)
                changeTemparature(temparature);
            p0 = density_function(temparature[blockIdx.x], cpre);
        }

        pickedId = pickedupIds[blockIdx.x];

        // if(threadIdx.x ==0)
            // printf("block: %d pickup: %d\n",blockIdx.x, pickedId );
            // fprintf( stderr,"threadIdx: %d, nTimes: %d\n", threadIdx.x, nt);
            //printf("threadIdx: %d, nTimes: %d\n", threadIdx.x, nt);
        __syncthreads();

        if(threadIdx.x == 0){
            pickedupIds[blockIdx.x] = get_int_random(room->objctNum);
            // printf("block: %d pickup: %d\n",blockIdx.x, pickedId );
        }


        secondChangeId = randomly_perturb(room, objsBlock, pickedId,
                        &sWrapper[0].wMask[maskStart], &sWrapper[0].backMask[maskStart], &costList[startId]);

        getWeightedCost(room, objsBlock, sWrapper[0].wPairRelation, &sWrapper[0].wmaskArea[2*blockIdx.x], &costList[startId]);
        // if(threadIdx.x == 0 && nt%10==0 ){
        //     for(int i=0; i<2; i++)
        //         printf("obj: %d, loc: %f, %f\n",i, objsBlock[i].translation[0], objsBlock[i].translation[1] );
            // displayResult(costList, weights);
        // }

        __syncthreads();

        cpost = sumUp_weighted_dataInShare(&costList[startId+1], weights, WEIGHT_NUM);

        costList[index] = 0;

        if(threadIdx.x == 0){
            p1 = density_function(temparature[blockIdx.x], cpost);
            alpha = fminf(1.0f, p1/p0);
            // printf("alpha: %f cpre: %f cpost: %f\n",alpha, cpre, cpost );
            if(alpha < THREADHOLD_T){
                restoreOrigin(room, &sWrapper[0].wMask[maskStart],&costList[startId],
                                &objsBlock[pickedId], nThreads);
                if(secondChangeId!=-1)
                    restoreOrigin(room, &sWrapper[0].wMask[maskStart],&costList[startId],
                                &objsBlock[secondChangeId], nThreads);
            }
            else if(cpost < cpre){
                getTemporalTransAndRot(room, objsBlock, sWrapper[0].resTransAndRot, cpost);
                cpre = cpost;
            }
        }//end thread 0
    }//end for
}

__global__
void Do_Metropolis_Hastings(sharedWrapper *gWrapper, float * gArray){
    sWrapper[0] = *gWrapper;
    if(blockIdx.x !=0 ){
        if(threadIdx.x < sWrapper[0].wRoom->objctNum){
            int objId = blockIdx.x * sWrapper[0].wRoom->objctNum + threadIdx.x;
            sWrapper[0].wObjs[objId] = sWrapper[0].wObjs[threadIdx.x];
        }
        int baseId = blockIdx.x *sWrapper[0].wRoom->mskCount;
        for(int i=threadIdx.x; i<sWrapper[0].wRoom->rowCount; i+=nThreads){
            for(int j=0;j<sWrapper[0].wRoom->colCount;j++)
                sWrapper[0].wMask[baseId + i * sWrapper[0].wRoom->colCount + j] = sWrapper[0].wMask[i * sWrapper[0].wRoom->colCount + j];
        }
    }

    float* costList = sWrapper[0].wFloats;
    float* temparature = (float *) & costList[nBlocks * nThreads];
    int * pickedupIds = (int*) &temparature[nBlocks];

	temparature[blockIdx.x] = get_float_random(10)/100;
    for(int i=threadIdx.x; i<gWrapper->nTimes; i+=nThreads)
        pickedupIds[i] = get_int_random(sWrapper[0].wRoom->objctNum);


    Metropolis_Hastings(costList, temparature, pickedupIds);
    // if(blockIdx.x == 0)
    // printf("thread: %d, err: \n",threadIdx.x, cudaGetLastError());
    __syncthreads();

    if(threadIdx.x < sWrapper[0].wRoom->objctNum){
        int singleSize = gWrapper->wRoom->objctNum * 4 + 1;
        for(int i=0; i<MAX_KEPT_RES; i++){
            gWrapper->resTransAndRot[singleSize * i] =  sWrapper[0].resTransAndRot[singleSize * i];

            int startPos = singleSize*i + 4*threadIdx.x;
            for(int k=1; k<5; k++)
                gWrapper->resTransAndRot[startPos + k] = sWrapper[0].resTransAndRot[startPos + k];
        }
    }
    __syncthreads();


    //gArray[threadIdx.x] = costList[threadIdx.x];
}

void generate_suggestions(Room * m_room, int nTimes){
    sharedWrapper *gWrapper;
    cudaMallocManaged(&gWrapper,  sizeof(sharedWrapper));

    gWrapper->nTimes = nTimes;
    cudaMallocManaged(&gWrapper->wRoom, sizeof(sharedRoom));
    m_room->CopyToSharedRoom(gWrapper->wRoom);

    int objMem = nBlocks *m_room->objctNum * sizeof(singleObj);
    cudaMallocManaged(&gWrapper->wObjs, objMem);
	for(int i=0; i<m_room->objctNum; i++)
		gWrapper->wObjs[i] = m_room->objects[i];

    int tMem = gWrapper->wRoom->mskCount * sizeof(unsigned char);
    cudaMallocManaged(&gWrapper->wMask, nBlocks *tMem);
    cudaMemcpy(gWrapper->wMask, m_room->furnitureMask, tMem, cudaMemcpyHostToDevice);
    cudaMallocManaged(&gWrapper->backMask, nBlocks *tMem);
    cudaMemset(gWrapper->backMask, 0, nBlocks*tMem);

	int floatMem =  (nBlocks *(2+nThreads)) * sizeof(float);
    cudaMallocManaged(&gWrapper->wFloats, floatMem);
    int maskAreaMem = 2*nBlocks * sizeof(float);
    cudaMallocManaged(&gWrapper->wmaskArea, maskAreaMem);
    cudaMemset(gWrapper->wmaskArea, 0, maskAreaMem);

    int pairMem = m_room->actualPairs.size() * 4 * sizeof(int);
    cudaMallocManaged(&gWrapper->wPairRelation, pairMem);
    for(int i=0;i<m_room->actualPairs.size();i++){
        for(int j=0;j<4;j++)
            gWrapper->wPairRelation[4*i+j]= m_room->actualPairs[i][j];
    }

    int resMem = (m_room->objctNum * 4 + 1) * MAX_KEPT_RES * sizeof(float);
    cudaMallocManaged(&gWrapper->resTransAndRot, resMem);

    float * gArray;
    cudaMallocManaged(&gArray, nThreads * sizeof(float));
    // cudaError_t err = cudaPeekAtLastError();
    // cout<<"error:"<<err<<endl;
	Do_Metropolis_Hastings<<<nBlocks, nThreads, sizeof(*gWrapper)>>>(gWrapper, gArray);
	cudaDeviceSynchronize();
    // err = cudaPeekAtLastError();
    // cout<<"error:"<<err<<endl;
    int singleSize = 4*m_room->objctNum + 1;
    for(int i=0, startId = 0; i< MAX_KEPT_RES; i++, startId = i*singleSize){
        cout<<"result: "<<i<<"- cost: "<<gWrapper->resTransAndRot[startId]<<endl;

        for(int n=0; n<m_room->objctNum; n++){
            cout<<"object: "<<n<<" pos and rot:";
            string res = "";
            for(int pi=1;pi<5;pi++)
                res+= to_string(gWrapper->resTransAndRot[startId+4*n+pi]) + " ";
            cout<<res<<endl;
        }
    }
    display_suggestions(m_room, gWrapper->resTransAndRot);
    cudaFree(gWrapper->wRoom);
    cudaFree(gWrapper->wObjs);
    cudaFree(gWrapper->wMask);
    cudaFree(gWrapper->backMask);
    cudaFree(gWrapper->wFloats);
    cudaFree(gWrapper->wPairRelation);
    cudaFree(gWrapper->resTransAndRot);
}

void startToProcess(Room * m_room, int nTimes){
    if(m_room->objctNum == 0)
        return;
	setUpDevices();

    clock_t start, finish;
    float costtime;
    start = clock();
	generate_suggestions(m_room, nTimes);

    finish = clock();
    costtime = (float)(finish - start) / CLOCKS_PER_SEC;
    cout<<"Runtime: "<<costtime<<endl;
}

int main(int argc, char** argv){
    //char* filename;
    int nTimes = DEFAULT_RUN_TIMES;
    for(int i=1; i<argc; i++){
        if(argv[i][0] == '-'){
            switch (argv[i][1]) {
                case 'n':
                    nTimes = (int)strtol(argv[i+1], (char **)nullptr, 10);
                    break;
            }
        }
    }
	//char* existance_file;
	//filename = new char[100];
	//existance_file = new char[100];
	//int r = strcpy_s(filename, 100, "E:/layoutParam.txt");
	//r = strcpy_s(existance_file, 100, "E:/fixedObj.txt");
	Room* parserRoom = new Room();
    setupDebugRoom(parserRoom);

	// parser_inputfile(filename, parserRoom);
	// parser_inputfile(existance_file, room, weights);
	// if (parserRoom != nullptr && (parserRoom->objctNum != 0 || parserRoom->wallNum != 0))
    startToProcess(parserRoom, nTimes);
	return 0;
}
