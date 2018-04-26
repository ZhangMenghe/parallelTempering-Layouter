#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "room.h"

using namespace std;
using namespace cv;

class layoutConstrains {
private:
	Room *room;
	vector<float> constrain_terms;

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
	layoutConstrains(Room* mr) {
		room = mr;
		constrain_terms = vector<float>(11, -1);
	}
	bool get_all_constrain_terms(vector<float> & params);

};


