#pragma once
#include <vector>
#include <map>
#include "predefinedConstrains.h"

using namespace std;

#ifndef __ROOM__
#define __ROOM__
struct mRect2f{
	float x,y;
	float width,height;
};
struct groupMapStruct{
	int gid;
	int memNum;
	float focal[3] = {INFINITY};
	int objIds[MAX_NUM_OBJS];
};
struct pairMapStruct{
	int pid;
	int objTypes[MAX_SUPPORT_TYPE] = {-1};
	int minDist[MAX_SUPPORT_TYPE];
	int maxDist[MAX_SUPPORT_TYPE];
};

struct singleObj{
	int id;
	bool isFixed;
	bool alignedTheWall;
	bool adjoinWall;
	mRect2f boundingBox;
	float vertices[8];
	float translation[3];
	float zrotation;
	float objWidth, objHeight;
	float zheight;
	int nearestWall;
	int catalogId;
	float area;
	float refRot;
	float refPos[2];

	unsigned char * objMask;
	int maskLen;

	float lastVertices[8];
	mRect2f lastBoundingBox;
	float lastTransAndRot[4];
};

struct wall{
	int id;
	float translation[3];
	float zrotation;
	float width;
	float a, b, c;//represent as ax+by+c=0
	float zheight;
	float vertices[4];
};
struct sharedRoom{
	int objctNum, wallNum, obstacleNum, freeObjNum;
	float half_width, half_height;
	float indepenFurArea, obstacleArea, wallArea;
	float overlappingThreshold;
	float RoomCenter[3];
	int freeObjIds[MAX_NUM_OBJS];
	int colCount, rowCount, mskCount;
	int pairNum, groupNum;
	groupMapStruct groupMap[MAX_GROUP_ALLOW];
	pairMapStruct pairMap[CONSTRAIN_PAIRS];
	wall deviceWalls[MAX_NUM_WALLS];
};
class Room{
private:
	unsigned char * furnitureMask_initial;

    //ax,ay,bx,by
	void init_a_wall(wall *newWall, vector<float> params);
    void init_an_object(vector<float>params, bool isFixed = false, bool isPrevious = false);
    void set_pairwise_map();

    void update_mask_by_wall(const wall* wal);
public:
    bool initialized;
	float center[3];
	vector<singleObj> objects;
	vector<wall> walls;
	singleObj * deviceObjs;
	wall * deviceWalls;
	groupMapStruct groupMap[MAX_GROUP_ALLOW];
	pairMapStruct pairMap[CONSTRAIN_PAIRS];
	int groupNum;
	int objctNum;
	int wallNum;
	int freeObjNum;
	float half_width;
	float half_height;
	int rowCount; int colCount;
	float indepenFurArea;
	float obstacleArea;
	float wallArea;
	float overlappingThreshold;
	int freeObjIds[MAX_NUM_OBJS];
	vector<vector<float>> obstacles;
	vector<vector<int>> actualPairs;
    Room() {
        center[0] = center[1] = center[2] =.0f;
        objctNum = 0;
        freeObjNum = 0;
        wallNum = 0;
        indepenFurArea = 0;
        obstacleArea = 0;
        initialized = false;

    }
    void initialize_room(float s_width = 800.0f, float s_height = 600.0f);
	void set_objs_pairwise_relation(const singleObj& obj1, const singleObj& obj2);
    void add_a_wall(vector<float> params);
    void add_an_object(vector<float> params, bool isPrevious = false, bool isFixed = false);
    void add_a_focal_point(vector<float> fp);
	void set_obj_zrotation(singleObj * obj, float new_rotation);
 	bool set_obj_translation(singleObj* obj, float tx, float ty);
    void add_an_obstacle(vector<float> params);
	void get_obstacle_vertices(float * vertices);
	void CopyToSharedRoom(sharedRoom *m_room);
};
#endif
