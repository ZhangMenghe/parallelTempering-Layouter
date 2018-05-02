#pragma once
#include "predefinedConstrains.h"
#include "room.cuh"
// #include "layoutConstrains.h"

#ifndef __LAYOUT__
#define __LAYOUT__

class automatedLayout{
	// layoutConstrains *constrains;
	float *weights;
	int debugParam = 0;
    __device__ __host__ void random_along_wall(int furnitureID);
public:
    // Room * room;
	float min_cost;
    float *resTransAndRot;
	automatedLayout(vector<float>in_weights);
	//void generate_suggestions();
	void display_suggestions();
	void initial_assignment(const Room* refRoom);
	__device__ float cost_function();
};

#endif
