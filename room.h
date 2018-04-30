#pragma once
#include <vector>
#include <map>
#include <utility>
#include "opencv2/core/core.hpp"
#include "predefinedConstrains.h"
using namespace std;
using namespace cv;
#ifndef __ROOM_H__
#define __ROOM_H__

struct mRect2f{
	float x,y;
	float width,height;
};

struct singleObj
{
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

struct wall
{
	int id;
	float translation[3];
	float zrotation;
	float width;
	float a, b, c;//represent as ax+by+c=0
	float zheight;
	float vertices[4];
};

class Room {
private:
	unsigned char ** furnitureMask_initial;
	Point card_to_graph_point(float x, float y) {
		return Point(int(half_width + x), int(half_height - y));
	}
	//ax,ay,bx,by
	void init_a_wall(wall *newWall, vector<float> params) {
		float ax = params[0], ay = params[1], bx = params[2], by = params[3];
		newWall->translation[0] = (ax + bx) / 2;newWall->translation[1] = (ay + by) / 2;newWall->translation[2] = 0;
		newWall->width = sqrtf(powf((by - ay), 2) + powf((bx - ax), 2));
		copy(params.begin(), params.end(), newWall->vertices);
		if (ax == bx) {
			newWall->zrotation = 0;
			newWall->b = 0; newWall->a = 1; newWall->c = -ax;
		}
		else if (ay == by) {
			newWall->zrotation = 90;
			newWall->a = 0; newWall->b = 1; newWall->c = -ay;
		}
		else {
			newWall->a = (by - ay) / (bx - ax); newWall->b = -1; newWall->c = -(newWall->a*ax - ay);
			newWall->zrotation = atanf(newWall->a)/ CV_PI *180;
		}
	}
	// 4*2 vertices, 2 center, 2 size, angle, label, zheight
	void init_an_object(vector<float>params, bool isFixed = false, bool isPrevious = false) {
		singleObj obj;
		obj.id = objects.size();
		//vertices
		copy(params.begin(), params.begin() + 8, obj.vertices);

		obj.translation[0] = params[8];obj.translation[0] =params[9];obj.translation[0] =.0f;
		obj.objWidth = params[10];
		obj.objHeight = params[11];

		obj.zrotation = params[12] * ANGLE_TO_RAD_F;
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
		//TODO:NEAREST WALL?
		//obj.nearestWall = find_nearest_wall(obj.translation[0], obj.translation[1]);

		objGroupMap[params[15]].push_back(obj.id);

		objects.push_back(obj);
		objctNum++;
		if (!isFixed)
			freeObjIds.push_back(obj.id);
		else
			//TODO: NO IDEAS HOW TO UPDATE MASK
			update_mask_by_object(&obj, furnitureMask_initial);//is a fixed object
	}

	void update_mask_by_wall(const wall* wal) {
		//TODO: DON'T KNOW HOW TO TACKLE WITH OBLIQUE WALL
	}

public:
	bool initialized;
	float center[3];
	vector<singleObj> objects;
	vector<wall> walls;
	map<int, vector<int>> objGroupMap;
	// map<int, vector<pair<int, Vec2f>>> pairMap;
	// map<int, Vec3f> focalPoint_map;
	int objctNum;
	int wallNum;
	unsigned char ** furnitureMask;
	float half_width;
	float half_height;
	float indepenFurArea;
	float obstacleArea;
	float wallArea;
	float overlappingThreshold;
	vector<int> freeObjIds;
	vector<vector<float>> obstacles;
	Room() {
		center[0] = center[1] = center[2] =.0f;
		objctNum = 0;
		wallNum = 0;
		indepenFurArea = 0;
		obstacleArea = 0;
		initialized = false;
	}

	void initialize_room(float s_width = 800.0f, float s_height = 600.0f) {
		initialized = true;
		half_width = s_width / 2;
		half_height = s_height / 2;
		overlappingThreshold = s_width * s_height * 0.005;
		// set_pairwise_map();
		int rowCount = int(s_height) + 1;	int colCount = int(s_width)+1;
		furnitureMask_initial = new unsigned char*[rowCount];
		for(int i=0; i< rowCount; i++){
			furnitureMask_initial[i] = new unsigned char[colCount];
			for(int j=0; j<colCount; j++)
				furnitureMask_initial[i][j] = 0;
		}
		furnitureMask = (unsigned char ** )malloc(sizeof(furnitureMask_initial));
		memcpy(furnitureMask, furnitureMask_initial, sizeof(furnitureMask_initial));
	}
	void add_a_wall(vector<float> params){
		wall newWall;
		newWall.id = walls.size();
		newWall.zheight = params[4];
		init_a_wall(&newWall, params);
		walls.push_back(newWall);
		wallNum++;
		if (fabs(fmod(newWall.zrotation, 90)) > 0.01)
			update_mask_by_wall(&newWall);
	}
	void add_an_object(vector<float> params, bool isPrevious = false, bool isFixed = false) {
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
	void update_mask_by_object(const singleObj* obj, unsigned char ** target, float movex = -1, float movey=-1){
	}
};
#endif
// void update_obj_boundingBox_and_vertices(singleObj& obj, float oldRot) {
// 	float s = sin(obj.zrotation - oldRot);
// 	float c = cos(obj.zrotation- oldRot);
// 	for (int i = 0; i < 4; i++)
// 		rot_around_point(obj.translation, obj.vertices[i], s, c);
// 	update_obj_boundingBox_by_vertices(obj);
//
// 	float offsetx = .0f, offsety = .0f;
//
// 	if (obj.boundingBox.x < -half_width)
// 		offsetx = -half_width - obj.boundingBox.x+1;
// 	else if (obj.boundingBox.x + obj.boundingBox.width > half_width)
// 		offsetx = half_width+1- obj.boundingBox.x - obj.boundingBox.width;
// 	if (obj.boundingBox.y > half_height)
// 		offsety = half_height- obj.boundingBox.y-1;
// 	else if (obj.boundingBox.y - obj.boundingBox.height < -half_height)
// 		offsety = -half_height - obj.boundingBox.y + obj.boundingBox.height+1;
// 	if (offsetx != 0 || offsety != 0) {
// 		obj.translation += Vec3f(offsetx, offsety, .0f);
// 		for (int i = 0; i < 4; i++) {
// 			obj.vertices[i] += Vec2f(offsetx, offsety);
// 		}
// 		obj.boundingBox.x += offsetx;
// 		obj.boundingBox.y += offsety;
// 	}
// }
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



// 	void rot_around_point(const Vec3f& center, Vec2f& pos, float s, float c) {
// 		// translate point back to origin:
// 		pos[0] -= center[0];
// 		pos[1] -= center[1];
//
// 		// rotate point
// 		float xnew = pos[0] * c - pos[1] * s;
// 		float ynew = pos[0] * s + pos[1] * c;
//
// 		// translate point back:
// 		pos[0] = xnew + center[0];
// 		pos[1] = ynew + center[1];
// 	}
// 	void set_obj_zrotation(float new_rotation, int id) {
// 		if (objects[id].zrotation == fmod(new_rotation, CV_PI))
// 			return;
// 		float old = objects[id].zrotation;
// 		objects[id].zrotation = fmod(new_rotation, CV_PI);
// 		update_obj_boundingBox_and_vertices(objects[id], old);
// 	}
//
// 	bool set_obj_translation(float tx, float ty, int id) {
// 		Mat_<uchar> tmpCanvas = furnitureMask;
// 		singleObj * obj = &objects[id];
//
// 		float movex = tx - objects[id].translation[0];
// 		float movey = ty - objects[id].translation[1];
// 		bool c1 = obj->boundingBox.x + movex <= -half_width;
// 		bool c2 = obj->boundingBox.x + obj->boundingBox.width + movex >= half_width;
// 		bool c3 = obj->boundingBox.y + movey >= half_height;
// 		bool c4 = obj->boundingBox.y - obj->boundingBox.height + movey <= -half_height;
// 		if (c1 || c2||c3 || c4)
// 			return false;
// 		update_mask_by_object(obj, tmpCanvas, movex, movey);
//
// 		if (cv::sum(furnitureMask)[0] + obj->area < cv::sum(tmpCanvas)[0])
// 			return false;
//
// 		obj->translation[0] = tx;
// 		obj->translation[1] = ty;
// 		for (int i = 0; i < 4; i++) {
// 			obj->vertices[i][0] += movex;
// 			obj->vertices[i][1] += movey;
// 		}
// 		obj->boundingBox.x += movex;
// 		obj->boundingBox.y += movey;
// 		return true;
// 	}
//
// 	void set_objs_rotation(vector<float> rotation) {
// 		for (int i = 0; i < objctNum; i++)
// 			objects[i].zrotation = rotation[i];
// 	}
//
// 	void set_pairwise_map() {
// 		vector<pair<int, Vec2f>> chair;
// 		// seat to seat
// 		chair.push_back(pair <int, Vec2f>(0, Vec2f(0, 50)));
// 		//coffee table to seat
// 		chair.push_back(pair <int, Vec2f>(1, Vec2f(40, 46)));
// 		//seat to end table
// 		chair.push_back(pair <int, Vec2f>(3, Vec2f(0, 30)));
//
// 		vector<pair<int, Vec2f>> bed;
// 		// bed TO nightstand
// 		bed.push_back(pair <int, Vec2f>(5, Vec2f(0, 30)));
// 		// bed to wall
// 		bed.push_back(pair <int, Vec2f>(100, Vec2f(0, 0)));
//
// 		vector<pair<int, Vec2f>> shelf;
// 		shelf.push_back(pair<int, Vec2f>(100, Vec2f(0, 0)));
//
// 		pairMap[TYPE_CHAIR] = chair;
// 		pairMap[TYPE_BED] = bed;
// 		pairMap[TYPE_SHELF] = shelf;
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
// 	float get_nearest_wall_dist(singleObj * obj) {
// 		float x = obj->translation[0], y = obj->translation[1];
// 		float min_dist = INFINITY, dist;
// 		//int min_id = -1;
// 		for (int i = 0; i < wallNum; i++) {
// 			dist = abs(walls[i].a * x + walls[i].b * y + walls[i].c) / sqrt(walls[i].a * walls[i].a + walls[i].b * walls[i].b);
// 			if (dist < min_dist) {
// 				min_dist = dist;
// 				obj->nearestWall = i;
// 			}
// 		}
// 		return min_dist;
// 	}
// 	void add_a_focal_point(vector<float> fp) {
// 		if(fp.size()>3)
// 			focalPoint_map[fp[3]] = Vec3f(fp[0], fp[1], fp[2]);
// 		else
// 			focalPoint_map[0] = Vec3f(fp[0], fp[1], fp[2]);
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
// 	void update_obj_boundingBox_by_vertices(singleObj& obj) {
// 		vector<float> xbox = { obj.vertices[0][0], obj.vertices[1][0], obj.vertices[2][0], obj.vertices[3][0] };
// 		vector<float> ybox = { obj.vertices[0][1], obj.vertices[1][1], obj.vertices[2][1], obj.vertices[3][1] };
//
// 		vector<float>::iterator it = min_element(xbox.begin(), xbox.end());
// 		//make sure bounding box start vertex is at the begining
// 		int startIdx = distance(xbox.begin(), it);
// 		vector<Vec2f> sub(obj.vertices.begin(), obj.vertices.begin() + startIdx);
// 		obj.vertices.insert(obj.vertices.end(), sub.begin(), sub.end());
// 		obj.vertices.erase(obj.vertices.begin(), obj.vertices.begin() + startIdx);
//
// 		float min_x = *it;
// 		float min_y = *min_element(ybox.begin(), ybox.end());
// 		float max_x = *max_element(xbox.begin(), xbox.end());
// 		float max_y = *max_element(ybox.begin(), ybox.end());
// 		obj.boundingBox = Rect2f(min_x, max_y, (max_x - min_x), (max_y - min_y));
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
