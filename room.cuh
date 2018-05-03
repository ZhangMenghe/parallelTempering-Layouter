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
struct groupMapStruct{
	int gid;
	int memNum;
	float focal[3] = {INFINITY, INFINITY, INFINITY};
	int objIds[MAX_NUM_OBJS];
};
struct pairMapStruct{
	int pid;
	int objTypes[MAX_SUPPORT_TYPE];
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

	float t(float d, float m, float M, int a = 2);
	void get_all_reflection(map<int, Vec3f> focalPoint_map, vector<Vec3f> &reflectTranslate, vector<float> & reflectZrot, float refk= INFINITY);
	void get_pairwise_relation(const singleObj& obj1, const singleObj& obj2, int&pfg, float&m, float&M, int & wallRelId);
	//Clearance :
	//Mcv(I) that minimize the overlap between furniture(with space)
	void cal_clearance_violation(float& mcv);
	//Circulation:
	//Mci support circulation through the room and access to all of the furniture.
	//Compute free configuration space of a person on the ground plane of the room
	//represent a person with radius = 18
	void cal_circulation_term(float& mci);
	//Pairwise relationships:
	//Mpd: for example  coffee table and seat
	//mpa: relative direction constraints
	void cal_pairwise_relationship(float& mpd, float& mpa);
	//Conversation
	//Mcd:group a collection of furniture items into a conversation area
	void cal_conversation_term(float& mcd, float& mca);
	//balance:
	//place the mean of the distribution of visual weight at the center of the composition
	void cal_balance_term(float &mvb);
	//Alignment:
	//compute furniture alignment term
	void cal_alignment_term(float& mfa, float&mwa);
	//Emphasis:
	//compute focal center
	void cal_emphasis_term(float& mef, float& msy, float gamma = 1);
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
	void freeMem();
    void RoomCopy(const Room & m_room);
    void initialize_room(float s_width = 800.0f, float s_height = 600.0f);
    void add_a_wall(vector<float> params);
    void add_an_object(vector<float> params, bool isPrevious = false, bool isFixed = false);
    void add_a_focal_point(vector<float> fp);
	__device__ __host__ void set_obj_zrotation(singleObj * obj, float new_rotation);
	__device__ __host__ bool set_obj_translation(singleObj* obj, float tx, float ty);
	__device__ float get_nearest_wall_dist(singleObj * obj);
	void update_mask_by_object(const singleObj* obj, unsigned char * target, float movex = -1, float movey=-1);
    void update_furniture_mask();

	// TODO:Calculate constrains
	__device__ void get_constrainTerms(float* constrains){
		for(int i=0;i<WEIGHT_NUM;i++)
			constrains[i] = 1.0f;
	}
};
#endif
