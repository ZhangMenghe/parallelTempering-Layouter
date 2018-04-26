#pragma once
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>
#include <math.h>  
using namespace cv;
using namespace std;

void graph_to_card(float half_width, float half_height, vector<Point2f> & rect) {
	for (vector<Point2f>::iterator it = rect.begin(); it != rect.end(); it++) {
		it->x -= half_width;
		it->y = half_height - it->y;
	}
}
void line_equation_2point(const Point2f &p1, const Point2f &p2, float& k, float&b) {
	//(y-y2)/(y1-y2) = (x-x2)/(x1-x2)
	k = (p2.y - p1.y) / (p2.x - p1.x);
	b = p1.y - k * p1.x;
}

Point2f point_proj_line(const Point2f &p, const float k, const float b) {
	float tx = (k*(p.y - b) + p.x) / (k*k + 1);
	return Point2f(tx, k*tx + b);
}
void get_bounding_xy(vector<Point2f> &r, const float cx, const int boundIdxStart, const float& k, const float& b, float* boundValue, int* boundIdx) {
	for (int i = 0; i < 4; i++) {
		// comapre project y 
		float projY = point_proj_line(r[i], k, b).y;
		if (projY > boundValue[3]) {
			boundValue[3] = projY;
			boundIdx[3] = boundIdxStart + i;
		}
		if (projY < boundValue[2]) {
			boundValue[2] = projY;
			boundIdx[2] = boundIdxStart + i;
		}
		//compare dist X
		float cdist = fabs(k * r[i].x - r[i].y + b);
		if (r[i].x > cx && cdist>boundValue[1]) {
			boundValue[1] = cdist;
			boundIdx[1] = boundIdxStart + i;
		}
		if (r[i].x < cx && cdist>boundValue[0]) {
			boundValue[0] = cdist;
			boundIdx[0] = boundIdxStart + i;
		}
	}
}
void get_bounding_vertices(const float * bound_ks, const float *bound_bs, vector<Point2f>& vertices) {
	int realIdx[4] = { 0,3,1,2 };
	for (int i = 0; i < 4; i++) {
		float interX = -(bound_bs[realIdx[(i + 1) % 4]] - bound_bs[realIdx[i]]) / (bound_ks[realIdx[(i + 1) % 4]] - bound_ks[realIdx[i]]);
		float interY = bound_ks[realIdx[i]] * interX + bound_bs[realIdx[i]];
		vertices.push_back(Point2f(interX, interY));
	}
}
void write_out_file(vector<vector<Point2f>> rects) {
	ofstream outfile;
	outfile.open("test.txt", std::fstream::out | std::fstream::app);
	if (outfile.is_open()) {
		outfile << "DEBUG_DRAW\t|\tvertices\n";
		for (vector<vector<Point2f>>::iterator rect = rects.begin(); rect != rects.end(); rect++) {
			for (vector<Point2f>::iterator point = rect->begin(); point != rect->end(); point++)
				outfile << "[" << to_string(point->x) << ", " << to_string(point->y) << "]\t|\t";
			outfile << "\n";
		}
	}
	outfile.close();
}
// input:  4 vertices, center, size, angle, label, zheight
vector<float> getUpdateInformationFromRotatedBox(const vector<Point2f>& rect){
	//position(3), float rot, float obj_width, float obj_height
	vector<float> res(14,0);
	for (int i = 0; i < 4; i++) {
		res[2 * i] = rect[i].x;
		res[2 * i + 1] = rect[i].y;
	}
	res[8] = (rect[0].x + rect[2].x) / 2;
	res[9] = (rect[0].y + rect[2].y) / 2;

	float dx01 = rect[0].x - rect[1].x; float dy01 = rect[0].y - rect[1].y;
	float dx12 = rect[1].x - rect[2].x; float dy12 = rect[1].y - rect[2].y;
	float dist1 = sqrt(pow(dx01,2) + pow(dy01,2));
	float dist2 = sqrt(pow(dx12, 2) + pow(dy12, 2));
	if (dist1 > dist2) {
		res[10] = dist1;
		res[11] = dist2;
		res[12] = atan2(dx01, dy01);
	}
	else {
		res[10] = dist2;
		res[11] = dist1;
		res[12] = atan2(dx12, dy12);
	}
	return res;
}

vector<vector<Point2f>> createRectFromParameters(vector<Point2f> rect1, vector<float>parameter) {
	vector<Point2f> rect2;
	vector<vector<Point2f>> res;
	for (int i = 0; i < 4; i++) 
		rect2.push_back(Point2f(parameter[2*i], parameter[2*i+1]));
	res.push_back(rect1);
	res.push_back(rect2);
	return res;
}
vector<vector<Point2f>> createRectFromParameters(vector<vector<float>> parameters, int id1, int id2) {
	vector<Point2f> rect1;
	for (int i = 0; i < 4; i++)
		rect1.push_back(Point2f(parameters[id1][2 * i], parameters[id1][2 * i + 1]));
	return createRectFromParameters(rect1, parameters[id2]);
}
vector<Point2f> merge2Obj(vector<vector<Point2f>> rectVector, float equal_thresh = 10.0f) {
	//vector<Point2f> rect1 = { Point2f(150,100), Point2f(350, 100), Point2f(350,200),Point2f(150,200) };
	//vector<Point2f> rect2 = { Point2f(400,200), Point2f(500,200), Point2f(500,300), Point2f(400,300) };
	vector<Point2f> res;
	vector<Point2f>* rect1 = &rectVector[0]; vector<Point2f> * rect2 = &rectVector[1];

	Point2f c1 = (rect1->at(0) + rect1->at(2)) / 2.0f;
	Point2f c2 = (rect2->at(0) + rect2->at(2)) / 2.0f;

	// construct line equation from c1 c2
	if (fabs(c1.x - c2.x)<equal_thresh || fabs(c1.y - c2.y)<equal_thresh) {
		float minX = std::min({ rect1->at(0).x, rect1->at(3).x, rect2->at(0).x, rect2->at(3).x });
		float maxX = std::max({ rect1->at(1).x, rect1->at(2).x, rect2->at(1).x, rect2->at(2).x });
		float minY = std::min({ rect1->at(0).x, rect1->at(1).x, rect2->at(0).x, rect2->at(1).x });
		float maxY = std::max({ rect1->at(2).x, rect1->at(3).x, rect2->at(2).x, rect2->at(3).x });
		res = { Point2f(minX, maxY), Point2f(maxX, maxY), Point2f(maxX, minY),Point2f(minX,minY) };
	}
	else {
		// minX, maxX, minY, maxY
		float boundValue[4] = { -INFINITY, -INFINITY, INFINITY,-INFINITY };
		int boundIdx[4] = { -1 };
		float pk; float pb;
		line_equation_2point(c1, c2, pk, pb);

		float vk = -1 / pk; float vb;
		//float vb = b = p.y + 1 / k * p.x;
		get_bounding_xy(*rect1, c1.x, 0, pk, pb, boundValue, boundIdx);
		get_bounding_xy(*rect2, c2.x, 4, pk, pb, boundValue, boundIdx);
		//get four bounding line equation
		Point2f pointBounding[4];
		for (int i = 0; i<4; i++)
			pointBounding[i] = rectVector[boundIdx[i] / 4][boundIdx[i] % 4];
		float bound_bs[4];
		float bound_ks[4] = { pk,pk,vk,vk };
		for (int i = 0; i < 4; i++)
			bound_bs[i] = pointBounding[i].y - bound_ks[i] * pointBounding[i].x;
		get_bounding_vertices(bound_ks, bound_bs, res);
	}
	rectVector.push_back(res);
	write_out_file(rectVector);
	return res;
}

// input:  4 vertices, center, size, angle, label, zheight
vector<float> mergeAgroup(const vector<vector<float>> &parameters, const vector<float>groupIds) {
	vector<Point2f> merged;
	float allHeights = 0;
	for (int i = 0; i < groupIds.size() - 1; i++) {
		if (merged.size() == 0)
			merged = merge2Obj(createRectFromParameters(parameters, groupIds[i], groupIds[i + 1]));
		else
			merged = merge2Obj(createRectFromParameters(merged, parameters[groupIds[i + 1]]));
		allHeights += parameters[groupIds[i + 1]][14];
	}
	vector<float> newInfos = getUpdateInformationFromRotatedBox(merged);
	vector<float>res(newInfos.begin(), newInfos.end());
	res.push_back(parameters[groupIds[0]][13]);
	res.push_back((allHeights + parameters[groupIds[0]][14]) / groupIds.size());
	return res;
}