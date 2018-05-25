
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
void cal_clearance_violation(sharedRoom * room, float maskArea, float& mcv){
    mcv = room->indepenFurArea - (maskArea-room->obstacleArea);
    mcv = (mcv < 0)? 0 : mcv;
	// printf("in clearance: %f\n",room->obstacleArea );
}
//Circulation:
//Mci support circulation through the room and access to all of the furniture.
__device__
void cal_circulation_term(float overlappingAreaPre, float overlappingAreaPost, float& mci){
    if(overlappingAreaPost == 0){
        mci = 0;
        return;
    }
    mci = (overlappingAreaPost - overlappingAreaPre) / overlappingAreaPost;
    //printf("a1: %f - a2: %f - mci: %f\n", overlappingAreaPre, overlappingAreaPost, mci);
}
//Pairwise relationships:
//Mpd: for example  coffee table and seat
//mpa: relative direction constraints
__device__
void cal_pairwise_relationship(sharedRoom*room, singleObj* objs, int * pairs,
                               int threadStride, float& mpd, float& mpa){
    singleObj * obj1, *obj2;
    float cosfg2;
    for(int i=threadIdx.x; i<room->pairNum; i+=threadStride){
        obj1 = &objs[pairs[4*i]];
        obj2 = &objs[pairs[4*i + 1]];
        mpd -= t(dist_between_points(obj1->translation, obj2->translation),
                pairs[4*i + 2], pairs[4*i + 3]);
        cosfg2 = powf((sinf(obj1->zrotation) * sinf(obj2->zrotation) + cosf(obj1->zrotation) * cosf(obj2->zrotation)), 2);
        mpa -= 8 * cosfg2*cosfg2 - 8 * cosfg2;
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
                    cosgf = cosf(obj2->zrotation) *(obj->translation[0] - obj2->translation[0])
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
void cal_balance_term(sharedRoom * room, const singleObj * obj, float &mvb){
    float centroid[3] = {.0f};
    centroid[0] += obj->area * obj->translation[0];
    centroid[1] += obj->area * obj->translation[1];
    centroid[2] += obj->area * obj->translation[2];
    centroid[0] /= room->indepenFurArea;centroid[1] /= room->indepenFurArea;centroid[2] /= room->indepenFurArea;
    //printf("%f - %f - %f- %f\n", centroid[0],centroid[1],centroid[2],sWrapper[0].wRoom->indepenFurArea);
    mvb = dist_between_points(centroid, room->RoomCenter);
}
//Alignment:
//compute furniture alignment term
__device__
void cal_alignment_term(sharedRoom * room, singleObj * objs, int gid, int mid, float& mfa, float&mwa){
    singleObj *obj1 = &objs[threadIdx.x] , *obj2;

    get_nearest_wall_dist(obj1,room->deviceWalls, room->wallNum);
    if(!obj1->adjoinWall)
        mwa -= cosf(4 * (obj1->zrotation
                - room->deviceWalls[obj1->nearestWall].zrotation
                - PI/2));

    for(int k = mid; k<room->groupMap[gid].memNum-1; k++){
        obj2 = &objs[threadIdx.x + k];
        mfa -= cosf(4 * (obj1->zrotation - obj2->zrotation));
    }
    //printf("mwa : %f  mfa: %f\n", mwa, mfa);
}
//Emphasis:
//compute focal center
__device__
void cal_emphasis_term(singleObj * obj, const float * focal, float& mef){
    if(focal[0] == INFINITY)
        return;
    get_obj_reflection(obj, focal);
    //printf("reflect: %f - %f - %f \n", obj->refPos[0], obj->refPos[1], obj->refRot);
    float dist = dist_between_points(focal, obj->translation);
    //printf("distance: %f\n",dist );
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
void displayResult(float * costList, float*weights){
    printf("emphasis: %f\n", costList[1]*weights[0] );
    printf("alignment: %f - %f\n", costList[2]*weights[1] , costList[3]*weights[2]  );
    printf("balance: %f\n", costList[4]*weights[3]  );
    printf("pairs: %f - %f\n", costList[5]*weights[4] , costList[6]*weights[5]  );
    printf("conversation: %f - %f\n",costList[7]*weights[6] , costList[8]*weights[7] );
	printf("emphasis2: %f\n", costList[9]*weights[8]  );
    printf("circulation: %f\n", costList[10]*weights[9]  );
    printf("clearance: %f\n", costList[11]*weights[10]  );
    printf("\n" );
}

__device__
void getWeightedCost(sharedRoom * room, singleObj* objs, int* pairRelations, float* maskArea, float* costList){
    costList[threadIdx.x] = 0;
    if(threadIdx.x < room->objctNum){

        singleObj * obj = &objs[threadIdx.x];
        for(int i=0; i<room->groupNum; i++){
            for(int j=0; j<room->groupMap[i].memNum; j++){
                if(room->groupMap[i].objIds[j] == threadIdx.x){
                    //printf("thread: %d , group:%d, memId:%d \n",threadIdx.x, i, j );
                    cal_emphasis_term(obj, room->groupMap[i].focal, costList[1]);
                    //printf("thread: %d, term: 1, cost: %f\n", threadIdx.x, costList[1]);
                    cal_alignment_term(room, objs, i, j,  costList[2], costList[3]);
                    //printf("thread: %d, term: 2,3, cost: %f, %f\n", threadIdx.x, costList[2], costList[3]);
                    cal_balance_term(room, obj, costList[4]);
                    //printf("thread: %d, term: 4, cost: %f\n", threadIdx.x, costList[4]);
                    cal_pairwise_relationship(room, objs, pairRelations,
                                              room->objctNum, costList[5], costList[6]);
                    //printf("thread: %d, term: 5,6, cost: %f, %f\n", threadIdx.x, costList[5], costList[6]);
                }
            }
        }

        //Each thread work on a group
        if(threadIdx.x < room->groupNum){
            cal_conversation_term(objs, room->groupMap[threadIdx.x].objIds,
                                    room->groupMap[threadIdx.x].memNum,
                                  costList[7], costList[8]);
            //printf("thread: %d, term: 7,8, cost: %f, %f\n", threadIdx.x, costList[7], costList[8]);
            cal_emphasis_term2(objs, &room->groupMap[threadIdx.x], costList[9]);
            //printf("thread: %d, term: 9, cost: %f\n", threadIdx.x, costList[9]);
        }
    }
    else if(threadIdx.x == room->objctNum){
        cal_circulation_term(maskArea[0], maskArea[1], costList[10]);
        //printf("circulation: %f\n",costList[10]);
        cal_clearance_violation(room, maskArea[0], costList[11]);
        // printf("clearance: %f\n", costList[11]);
    }
}
