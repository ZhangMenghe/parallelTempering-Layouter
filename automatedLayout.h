#pragma once
#include "layoutConstrains.h"
#include "room.h"
class automatedLayout
{
private:
	layoutConstrains *constrains;
	float *weights;
	void setUpDevices();
    void random_along_wall(int furnitureID);
    float cost_function();
public:
    Room * room;
	float min_cost;
    float **resTransAndRot;
	automatedLayout(Room* m_room, vector<float>in_weights);
	void generate_suggestions();
	void display_suggestions();
};
