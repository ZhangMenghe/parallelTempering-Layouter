#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "layoutConstrains.h"
#include "room.h"
#include <queue>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

using namespace cv;

class automatedLayout
{
private:
	layoutConstrains *constrains;
	Room * room;
	
	vector<float>weights;
	float cost_function();
	float density_function(float cost);
	void randomly_perturb(vector<Vec3f>& ori_trans, vector<float>& ori_rot, vector<int>& selectedid, int flag);
	void Metropolis_Hastings(int * pickedIdxs,unsigned int seed);
	void random_translation(int furnitureID, default_random_engine generator);
	void random_along_wall(int furnitureID);
	void initial_assignment();
	void setUpDevices();
public:

	int resNum = 3;
	float min_cost;
	// queue<vector<Vec3f>> res_transform;
	// queue<vector<float>> res_rotation;


	automatedLayout(Room* m_room, vector<float>in_weights);
	void generate_suggestions();
	void display_suggestions();
};


//void do_authoring(vector<singleObj>& objs);
