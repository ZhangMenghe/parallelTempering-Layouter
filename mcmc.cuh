#pragma once
#include <vector>
#include <map>
#include <utility>
#include "opencv2/core/core.hpp"
#include "predefinedConstrains.h"
// #include "layoutConstrains.h"
using namespace std;
using namespace cv;

class Room;
class automatedLayout;

#ifndef __CUHEAD__
#define __CUHEAD__
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

class automatedLayout{
	// layoutConstrains *constrains;
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
    void update_mask_by_object(const singleObj* obj, unsigned char * target, float movex = -1, float movey=-1);
    void update_furniture_mask();
};
#endif
