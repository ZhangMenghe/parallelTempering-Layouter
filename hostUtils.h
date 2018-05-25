#include <string>
#include <fstream>
#include "room.cuh"

__device__ __managed__ float weights[11]={1.0f};
void setupDebugRoom(Room* room){
    float wallParam1[] = {-200, 150, 200, 150};
    float wallParam2[] = {-200, -150, 200, -150};
    float wallParam3[] = {-200, -150, -200, 150};
    float wallParam4[] = {200, -150, 200, 150};
    float objParam[] = {0, 0, 50, 50, 0, 0, 10};
    float bedParam[] = {0, 0, 100, 200, 0, 4, 10};
    float deskParam[] = {0, 0, 40, 100, 0, 7, 10};
    float obsParam[] = {-50,50,50,50,50,-50,-50,-50};
    float fpParam[] = {0, 150, 0};
    float mWeights[] = {1.0f, 0.01f, 0.01f, 0.1f, 1.0f, 1.0f, 1.0f, 0.001f, 0.01f, 1.0f, 2.0f};

    room->initialize_room(400.0f, 300.0f);
    room->add_a_wall(vector<float>(wallParam1,wallParam1 + 4));
    room->add_a_wall(vector<float>(wallParam2,wallParam2 + 4));
    room->add_a_wall(vector<float>(wallParam3,wallParam3 + 4));
    room->add_a_wall(vector<float>(wallParam4,wallParam4 + 4));
    room->add_an_object(vector<float>(objParam,objParam + 7), false,true);
    room->add_an_object(vector<float>(objParam,objParam + 7));
    room->add_an_object(vector<float>(bedParam,bedParam + 7));
    room->add_an_object(vector<float>(deskParam,deskParam + 7));
    room->add_a_focal_point(vector<float>(fpParam,fpParam + 3));
    // room->add_an_obstacle(vector<float>(obsParam,obsParam + 8));
    room->objects[1].adjoinWall = true;
    for(int i=0;i<11;i++)
        weights[i] = mWeights[i];
    for(int i=0; i< room->objctNum-1; i++){
        for(int j=i+1; j<room->objctNum; j++)
            room->set_objs_pairwise_relation(room->objects[i], room->objects[j]);
    }

}
void parser_inputfile(const char* filename, Room * parser_inputfile) {
	ifstream instream(filename);
	string str;
	vector<vector<float>> parameters;
	vector<char> cateType;
	char  delims[] = " :,\t\n";
	char* context = nullptr;
	while (instream && getline(instream, str)) {
		if (!str.length())
			continue;
		char * charline = new char[300];
		int r = strcpy_s(charline, 300, str.c_str());
		char * itemCate = strtok_s(charline,delims,&context);
		vector<float>param;
		char * token = strtok_s(nullptr, delims, &context);
		while (token != nullptr) {
			param.push_back(atof(token));
			token = strtok_s(nullptr, delims, &context);
		}
		parameters.push_back(param);
		cateType.push_back(itemCate[0]);
	}
	instream.close();
	int itemNum = cateType.size();
	vector<vector<float>> fixedObjParams;
	vector<vector<float>> mergedObjParams;
	vector<int> groupedIds;
	int startId = 0;
	if (cateType[0] == 'r') {
		parser_inputfile->initialize_room(parameters[0][0], parameters[0][1]);
		startId = 1;
	}
	else if(!parser_inputfile->initialized)
		parser_inputfile->initialize_room();
	for (int i = startId; i < itemNum; i++) {
		switch (cateType[i])
		{
		case '#':
			break;
		//add a new wall
		case 'w':
			parser_inputfile->add_a_wall(parameters[i]);
			break;
		case 'f':
			parser_inputfile->add_an_object(parameters[i]);
			break;
		case 'p':
			parser_inputfile->add_a_focal_point(parameters[i]);
			break;
		case 'v':
            for(int k=0;k<parameters[i].size(); k++)
                weights[k] = parameters[i][k];
			break;
        default:
            break;
        }
    }
}

string getPureNumFromVector(float * vertices, int length) {
	string res = "";
	for (int i = 0; i < length; i++)
		res += to_string(vertices[i]) + " ";
	return res;
}

void display_suggestions(Room* room, float *resTransAndRot) {
    ofstream outfile;
	outfile.open("E:/recommendation.txt", 'w');
	outfile << "RoomSize: "<<to_string(int(room->half_width*2))<<" "<<to_string(int(room->half_height*2))<<"\r\n";
	if (outfile.is_open()) {
		outfile << "WALL_Id zheight vertices zrotation\r\n";
		for (int i = 0; i < room->wallNum; i++) {
			wall * tmp = &room->walls[i];
			outfile << to_string(tmp->id) << " " <<tmp->zheight<<" "<< getPureNumFromVector (tmp->vertices, 4)<<tmp->zrotation<<"\r\n";
		}

        int singleSize = 4*room->objctNum + 1;
		for (int i = 0; i < room->objctNum; i++) {
			outfile << "FURNITURE_Id Category Height ObjWidth ObjHeight\r\n";
			singleObj *tmp = &room->objects[i];
			outfile << tmp->id << " " << tmp->catalogId  << " "<<tmp->zheight<< " "<<tmp->objWidth<< " "<<tmp->objHeight;

			outfile << "\r\n";
            int objoffset = 4*i+1;
			for (int res=0, startId=0; res<MAX_KEPT_RES; res++, startId=res*singleSize)
                outfile << "Recommendation" << res << " " << getPureNumFromVector(&resTransAndRot[startId+objoffset], 4)<< "\r\n";
		}
		outfile << "Obstacle Vertices\r\n";
		string obstacleContent = "";
		for (int i = 0; i < room->obstacles.size(); i++) {

			for (int j = 0; j < 8; j++)
				obstacleContent += to_string(room->obstacles[i][j]) + " ";
			obstacleContent += "\r\n";
		}
		outfile << obstacleContent;
		outfile << "Point_Focal Position\r\n";
        for(int i=0;i<room->groupNum;i++){
            groupMapStruct * gmap = & room->groupMap[i];
            if(gmap->focal[0]!=INFINITY)
                outfile<<getPureNumFromVector(gmap->focal, 3)<<endl;
        }
		outfile.close();
	}
	else
		exit(-1);

}
