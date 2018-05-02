class automatedLayout;
void automatedLayout::random_along_wall(int furnitureID){

}
__device__
float automatedLayout::cost_function(){
	return 0;
}
void automatedLayout::initial_assignment(const Room * refRoom){
	for (int i = 0; i < refRoom->freeObjNum; i++) {
		singleObj* obj = &room->deviceObjs[refRoom->freeObjIds[i]];
		if (obj->adjoinWall)
			random_along_wall(refRoom->freeObjIds[i]);
		else if (obj->alignedTheWall)
			cout<<"do nothing now"<<endl;
			// room->set_obj_zrotation(refRoom->walls[rand() % refRoom->wallNum].zrotation, room->freeObjIds[i]);
	}
	room->update_furniture_mask();
}
