#pragma once
#include <vector>
#include <map>
#include "predefinedConstrains.h"
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

#ifndef __ROOM__
#define __ROOM__
struct mRect2f{
	float x,y;
	float width,height;
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
class Room{
private:
	unsigned char * furnitureMask_initial;
	Point card_to_graph_point(float x, float y) {
		return Point(int(half_width + x), int(half_height - y));
	}
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
	map<int, vector<int>> objGroupMap;
	map<int, vector<pair<int, Vec2f>>> pairMap;
	map<int, vector<float>> focalPoint_map;
	int objctNum;
	int wallNum;
	int freeObjNum;
	unsigned char * furnitureMask;
	float half_width;
	float half_height;
	int rowCount; int colCount;
	float indepenFurArea;
	float obstacleArea;
	float wallArea;
	float overlappingThreshold;
	int freeObjIds[MAX_NUM_OBJS];
	vector<vector<float>> obstacles;
    Room() {
        center[0] = center[1] = center[2] =.0f;
        objctNum = 0;
        freeObjNum = 0;
        wallNum = 0;
        indepenFurArea = 0;
        obstacleArea = 0;
        initialized = false;

    }
    void RoomCopy(const Room & m_room);
    void initialize_room(float s_width = 800.0f, float s_height = 600.0f);
    void add_a_wall(vector<float> params);
    void add_an_object(vector<float> params, bool isPrevious = false, bool isFixed = false);
    void add_a_focal_point(vector<float> fp);
	__device__ __host__ void set_obj_zrotation(singleObj * obj, float new_rotation);
	__host__ __device__ void update_obj_boundingBox_and_vertices(singleObj* obj);
	void update_mask_by_object(const singleObj* obj, unsigned char * target, float movex = -1, float movey=-1);
    void update_furniture_mask();
};
#endif