#include<iostream>
#include "room.cuh"
#include "utils.cuh"
#include "math.h"

using namespace std;

//ax,ay,bx,by
void Room::init_a_wall(wall *newWall, vector<float> params) {
	float ax = params[0], ay = params[1], bx = params[2], by = params[3];
	newWall->translation[0] = (ax + bx) / 2;newWall->translation[1] = (ay + by) / 2;newWall->translation[2] = 0;
	newWall->width = sqrtf(powf((by - ay), 2) + powf((bx - ax), 2));
	copy(params.begin(), params.end(), newWall->vertices);
	if (ax == bx) {
		newWall->zrotation = 0;
		newWall->b = 0; newWall->a = 1; newWall->c = -ax;
	}
	else if (ay == by) {
		newWall->zrotation = PI/2;
		newWall->a = 0; newWall->b = 1; newWall->c = -ay;
	}
	else {
		newWall->a = (by - ay) / (bx - ax); newWall->b = -1; newWall->c = -(newWall->a*ax - ay);
		newWall->zrotation = atanf(newWall->a)/ PI;
	}
}
// 4*2 vertices, 2 center, 2 size, angle, label, zheight
void Room::init_an_object(vector<float>params, bool isFixed, bool isPrevious) {
	singleObj obj;
	obj.id = objects.size();
	//vertices
	copy(params.begin(), params.begin() + 8, obj.vertices);

	obj.translation[0] = params[8];obj.translation[1] =params[9];obj.translation[2] =.0f;
	obj.objWidth = params[10];
	obj.objHeight = params[11];
	set_obj_zrotation(&obj, params[12] * ANGLE_TO_RAD_F);
	cout<<"boundingBox: "<<obj.boundingBox.y<< " "<< obj.boundingBox.height<<endl;
	//obj.zrotation = params[12] * ANGLE_TO_RAD_F;
	obj.catalogId = params[13];
	obj.zheight = params[14];
	obj.area = obj.objWidth * obj.objHeight;
	obj.isFixed = isFixed;
	obj.alignedTheWall = (obj.catalogId == TYPE_SHELF || obj.catalogId == TYPE_BED || obj.catalogId == TYPE_TABLE) ? true : false;
	obj.adjoinWall = (obj.catalogId == TYPE_SHELF || obj.catalogId == TYPE_BED || obj.catalogId == TYPE_TABLE) ? true : false;

	// TODO: is it necessary?
	// if (!isPrevious)//existing objs' values should be
		// update_obj_boundingBox_and_vertices(obj, 0);

	indepenFurArea += obj.objWidth * obj.objHeight; //get_single_obj_maskArea(obj.vertices);

	int gidx = 0;
	for(; gidx<groupNum; gidx++){
		if(groupMap[gidx].gid == params[15])
			groupMap[gidx].objIds[groupMap[gidx].memNum++] = obj.id;
	}
	if(gidx == groupNum){
		groupMap[groupNum].gid = params[15];
		groupMap[groupNum].memNum = 1;
		groupMap[groupNum].objIds[0] = obj.id;
		groupNum++;
	}

	objects.push_back(obj);
	objctNum++;
	if (!isFixed)
		freeObjIds[freeObjNum++] = obj.id;
}
void Room::set_pairwise_map() {
	pairMap[0].pid = TYPE_CHAIR;
	int mtype[3] = {TYPE_CHAIR, TYPE_COFFETABLE, TYPE_ENDTABLE};
	copy(begin(mtype), end(mtype), begin(pairMap[0].objTypes));
	int mdist[3] = {0,40,0}; int mdistm[3] =  {50,46,30};
	copy(begin(mdist), end(mdist), begin(pairMap[0].minDist));
	copy(begin(mdistm), end(mdistm), begin(pairMap[0].maxDist));

	pairMap[1].pid = TYPE_BED;
	pairMap[1].objTypes[0] = TYPE_NIGHTSTAND;
	pairMap[1].minDist[0] = 0;
	pairMap[1].maxDist[0] = 30;
}
void Room::set_objs_pairwise_relation(const singleObj& obj1, const singleObj& obj2){
	const singleObj* indexObj = (obj1.catalogId <= obj2.catalogId)?&obj1:&obj2;
	const singleObj* compObj = (obj1.id == indexObj->id)? &obj2:&obj1;
	for(int i=0; i<CONSTRAIN_PAIRS; i++){
		if(indexObj->catalogId == pairMap[i].pid){
			for(int j=0; pairMap[i].objTypes[j]!=-1&&j<MAX_SUPPORT_TYPE; j++){
				if(pairMap[i].objTypes[j] == compObj->catalogId){
					vector<int> pair{indexObj->id, compObj->id, pairMap[i].minDist[j],  pairMap[i].maxDist[j]};
					actualPairs.push_back(pair);
					break;
				}
			}
		}
	}
}
void Room::update_mask_by_wall(const wall* wal) {
	//TODO: DON'T KNOW HOW TO TACKLE WITH OBLIQUE WALL
}



void Room::CopyToSharedRoom(sharedRoom *m_room){
	m_room->objctNum = objctNum;
	m_room->wallNum = wallNum;
	m_room->freeObjNum = freeObjNum;
	m_room->half_width = half_width;
	m_room->half_height = half_height;
	m_room->indepenFurArea = indepenFurArea;
	m_room->obstacleArea = obstacleArea;
	m_room->wallArea = wallArea;
	m_room->overlappingThreshold = overlappingThreshold;
	m_room->colCount = colCount;
	m_room->rowCount = rowCount;
	m_room->mskCount = colCount * rowCount;
	m_room->pairNum = actualPairs.size();
	m_room->groupNum = groupNum;
	m_room->RoomCenter[0] = center[0];m_room->RoomCenter[1] = center[1];m_room->RoomCenter[2] = center[2];
	cudaMemcpy(m_room->freeObjIds, freeObjIds, freeObjNum* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_room->groupMap, groupMap, MAX_GROUP_ALLOW* sizeof(groupMapStruct), cudaMemcpyHostToDevice);
	cudaMemcpy(m_room->pairMap, pairMap, CONSTRAIN_PAIRS* sizeof(pairMapStruct), cudaMemcpyHostToDevice);
	for(int i=0;i<wallNum;i++)
		m_room->deviceWalls[i] = walls[i];
	//
	//
	// int tMem = colCount*rowCount * sizeof(unsigned char);
	// cudaMallocManaged(&furnitureMask, tMem);
	// cudaMallocManaged(&furnitureMask_initial, tMem);
	// cudaMemcpy(furnitureMask, m_room->furnitureMask, tMem, cudaMemcpyHostToDevice);
	// cudaMemcpy(furnitureMask_initial, m_room->furnitureMask_initial, tMem, cudaMemcpyHostToDevice);
	// //TODO:obstacle
}

void Room::initialize_room(float s_width, float s_height) {
	initialized = true;
	groupNum = 0;
	half_width = s_width / 2;
	half_height = s_height / 2;
	overlappingThreshold = s_width * s_height * 0.005;
	set_pairwise_map();
	rowCount = int(s_height) + 1;	colCount = int(s_width)+1;
	int tMem = rowCount * colCount * sizeof(unsigned char);
	furnitureMask_initial = (unsigned char *)malloc(tMem);
	memset(furnitureMask_initial, (unsigned char)0, colCount*rowCount);
	furnitureMask = (unsigned char* )malloc(tMem);
	memset(furnitureMask,(unsigned char)0 , colCount*rowCount);
	// cout<<int(furnitureMask[100])<<"asdfasdf"<<endl;
}
void Room::add_a_wall(vector<float> params){
	wall newWall;
	newWall.id = walls.size();
	newWall.zheight = params[4];
	init_a_wall(&newWall, params);
	walls.push_back(newWall);
	wallNum++;
	if (fabs(fmod(newWall.zrotation, PI)) > 0.01)
		update_mask_by_wall(&newWall);
}
void Room::add_an_object(vector<float> params, bool isPrevious, bool isFixed) {
	if (params.size() < 15) {
		float hw = params[2] / 2, hh = params[3] / 2;
		float cx = params[0], cy = params[1];
		float res[8] = { -hw + cx, hh + cy, hw + cx, hh + cy, hw + cx, -hh + cy, -hw + cx, -hh + cy };
		vector<float>vertices(res, res + 8);// get_vertices_by_pos(params[0], params[1], params[2] / 2, params[3] / 2);
		params.insert(params.begin(), vertices.begin(), vertices.end());
	}
	if (isPrevious) {
		switch (int(params[13]))
		{
		case 1:
			params[13] = TYPE_FLOOR;
			break;
		case 3://chair
			params[13] = TYPE_CHAIR;
			break;
		case 8:
			params[13] = TYPE_WALL;
			break;
		case 10:
			params[13] = TYPE_OTHER;
			break;
		case 11:
			params[13] = TYPE_CEILING;
			break;
		}
	}
	//default groupid is 0
	if(params.size()<16)
		params.push_back(0);

	init_an_object(params, isFixed, isPrevious);
}
void Room::add_a_focal_point(vector<float> fp) {
	int groupId = (fp.size() == 3)? 0:fp[3];
	for(int i=0; i<groupNum; i++){
		if(groupId == groupMap[i].gid)
			copy(fp.begin(), fp.begin()+3, groupMap[i].focal);
	}
}


void Room::set_obj_zrotation(singleObj * obj, float nrot) {
	float oldRot = obj->zrotation;
	nrot = remainderf(nrot, 2*PI);
	obj->zrotation = nrot;
	float gap = obj->zrotation - oldRot;
	float s = sinf(gap); float c=cosf(gap);
	float minx = INFINITY,maxx =-INFINITY, miny=INFINITY, maxy = -INFINITY;
	for(int i=0; i<4; i++){
		rot_around_point(obj->translation, &obj->vertices[2*i], &obj->vertices[2*i+1], s, c);
		minx = (obj->vertices[2*i] < minx)? obj->vertices[2*i]:minx;
		maxx = (obj->vertices[2*i] > maxx)? obj->vertices[2*i]:maxx;
		miny = (obj->vertices[2*i + 1] < miny)? obj->vertices[2*i+1]:miny;
		maxy = (obj->vertices[2*i + 1] > maxy)? obj->vertices[2*i+1]:maxy;
	}
	obj->boundingBox.x = minx; obj->boundingBox.y=maxy;
	obj->boundingBox.width = maxx-minx; obj->boundingBox.height = maxy-miny;
}

__device__
float Room::get_nearest_wall_dist(singleObj * obj) {
	float x = obj->translation[0], y = obj->translation[1];
	float min_dist = INFINITY, dist;

	for (int i = 0; i < wallNum; i++) {
		dist = fabsf(deviceWalls[i].a * x + deviceWalls[i].b * y + deviceWalls[i].c) / sqrtf(deviceWalls[i].a * deviceWalls[i].a + deviceWalls[i].b * deviceWalls[i].b);
		if (dist < min_dist) {
			min_dist = dist;
			obj->nearestWall = i;
		}
	}
	// printf("%d : %f\n", obj->nearestWall, min_dist);
	return min_dist;
}
void Room::update_mask_by_object(const singleObj* obj, unsigned char * target, float movex, float movey){
}


// 	float get_single_obj_maskArea(vector<Vec2f> vertices) {
// 		vector<vector<Point>> contours;
// 		vector<Point> contour;
// 		for (int n = 0; n < 4; n++)
// 			contour.push_back(card_to_graph_point(vertices[n][0], vertices[n][1]));
// 		contours.push_back(contour);
// 		Mat_<uchar> canvas = Mat::zeros(half_height*2, half_width*2, CV_8UC1);
// 		drawContours(canvas, contours, -1, 1, FILLED, 8);
// 		return cv::sum(canvas)[0];
// 	}


// 	void update_mask_by_object(const singleObj* obj, Mat_<uchar> & target, float movex = -1, float movey=-1) {
// 		vector<Point> contour;
// 		vector<vector<Point>> contours;
// 		for (int i = 0; i < 4; i++)
// 			contour.push_back(card_to_graph_point(obj->vertices[i][0], obj->vertices[i][1]));
// 		contours.push_back(contour);
// 		if (movex != -1) {
// 			drawContours(target, contours, 0, 0, FILLED, 8);
// 			vector<Point> contour2;
// 			for (int i = 0; i < 4; i++)
// 				contour2.push_back(card_to_graph_point(movex + obj->vertices[i][0], movey + obj->vertices[i][1]));
// 			contours.push_back(contour2);
// 			drawContours(target, contours, 1, 1, FILLED, 8);
// 		}
// 		else
// 			drawContours(target, contours,-1, 1, FILLED, 8);
// 	}

// 	float * get_objs_TransAndRot(){
// 		int FloatSize = sizeof(float);
// 		int singleItemSize = 4 * FloatSize;
// 		float * res = (float *)malloc(objctNum * singleItemSize);
// 		for (int i = 0; i < objctNum; i++){
// 			int startPos = i*singleItemSize;
// 			res[startPos] = objects[i].translation[0];
// 			res[startPos +   FloatSize] = objects[i].translation[1];
// 			res[startPos + 2*FloatSize] = objects[i].translation[2];
// 			res[startPos + 3*FloatSize] = objects[i].zrotation;
// 		}
// 		return res;
// 	}


//



//
// 	void add_an_obstacle(vector<float> vertices) {
// 		vector<Point> contour;
// 		vector<vector<Point>> contours;
//
// 		for (int i = 0; i < 4; i++)
// 			contour.push_back(card_to_graph_point(vertices[2 * i], vertices[2 * i + 1]));
// 		contours.push_back(contour);
// 		drawContours(furnitureMask_initial, contours, -1, 1, FILLED, 8);
// 		obstacleArea = cv::sum(furnitureMask_initial)[0];
// 		//cout << "obstacleArea:  " << obstacleArea<<endl;
// 		obstacles.push_back(vertices);
// 	}


//
// 	void update_furniture_mask() {
// 		furnitureMask = furnitureMask_initial.clone();
// 		float test1 = cv::sum(furnitureMask)[0];
// 		vector<vector<Point>> contours;
// 		for (int i = 0; i < objctNum; i++) {
// 			singleObj * obj = &objects[i];
// 			vector<Point> contour;
// 			for (int n = 0; n < 4; n++)
// 				contour.push_back(card_to_graph_point(obj->vertices[n][0], obj->vertices[n][1]));
// 			contours.push_back(contour);
// 		}
// 		//drawContours(furnitureMask, contours, 0, 1, FILLED, 8);
// 		drawContours(furnitureMask, contours, -1, 1, FILLED, 8);
// 		float test = cv::sum(furnitureMask)[0];
// 	}
// 	void change_obj_freeState(singleObj* obj) {
// 		if (obj->isFixed)
// 			freeObjIds.erase(remove(freeObjIds.begin(), freeObjIds.end(), obj->id));
// 		else
// 			freeObjIds.push_back(obj->id);
// 		obj->isFixed = !obj->isFixed;
// 	}
// };
