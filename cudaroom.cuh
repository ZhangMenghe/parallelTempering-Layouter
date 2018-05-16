#include <stdio.h>
#include "utils.cuh"
__device__
bool set_obj_translation(sharedRoom * room, singleObj* obj, float cx, float cy){
    cx = (get_int_random(2) == 0)?cx:-cx;
    cy = (get_int_random(2) == 0)?cy:-cy;
    float halfw = obj->boundingBox.width/2, halfh = obj->boundingBox.height/2;
    if( cx + halfw > room->half_width || cx-halfw < -room->half_width
     || cy + halfh > room->half_height || cy-halfh < -room->half_height)
     return false;

    float movex = cx - obj->translation[0], movey = cy-obj->translation[1];
    obj->translation[0] = cx; obj->translation[1]=cy;
    for(int i=0; i<4; i++){
        obj->vertices[2*i]+=movex;
        obj->vertices[2*i + 1] += movey;
    }
    obj->boundingBox.x += movex; obj->boundingBox.y += movey;
    return true;
}
__device__
void set_obj_zrotation(singleObj * obj, float nrot) {
	float oldRot = obj->zrotation;
	nrot = remainderf(nrot, 2*PI);
	obj->zrotation = nrot;
	float gap = obj->zrotation - oldRot;
	float s = sinf(gap); float c=cosf(gap);
	float minx = INFINITY, maxx =-INFINITY, miny=INFINITY, maxy = -INFINITY;
	for(int i=0; i<4; i++){
		rot_around_a_point(obj->translation, &obj->vertices[2*i], &obj->vertices[2*i+1], s, c);
		minx = (obj->vertices[2*i] < minx)? obj->vertices[2*i]:minx;
		maxx = (obj->vertices[2*i] > maxx)? obj->vertices[2*i]:maxx;
		miny = (obj->vertices[2*i + 1] < miny)? obj->vertices[2*i+1]:miny;
		maxy = (obj->vertices[2*i + 1] > maxy)? obj->vertices[2*i+1]:maxy;
	}
	obj->boundingBox.x = minx; obj->boundingBox.y=maxy;
	obj->boundingBox.width = maxx-minx; obj->boundingBox.height = maxy-miny;
}
//update_mask_by_boundingBox(backupMask, rect, room->rowCount/2, room->colCount, threadIdx.x, nThreads, 1);
__device__
void update_mask_by_boundingBox(unsigned char* mask, mRect2f boundingBox, int halfRowNum, int colNum, int absThreadIdx, int threadStride, int addition=1){
    for(int y = boundingBox.y - absThreadIdx; y > boundingBox.y - boundingBox.height; y -= threadStride){
        for(int x=boundingBox.x; x<boundingBox.x + boundingBox.width; x++)
            mask[(halfRowNum - y) *colNum  + x] += addition;
    }
}
__device__
void update_mask_by_object(unsigned char* mask, float* tmpSlot, float * vertices,
                            mRect2f boundingBox, int halfRowNum, int colNum, int absThreadIdx, int threadStride, int addition=1){

    int boundX = boundingBox.x + boundingBox.width;
    //Ideally, each thread process a row
    for(int y = boundingBox.y - absThreadIdx; y > boundingBox.y - boundingBox.height; y -= threadStride){
        //int test = 0;
        for(int x = boundingBox.x; x<boundX; x++){
            if(!point_in_rectangle(tmpSlot, vertices, x, y));
            else{
                int endIndx = binary_search_Inside_Point(x, boundX - 1, 0, y, tmpSlot, vertices);
                while(x <= endIndx){
                    mask[(halfRowNum - y) *colNum  + x] += addition;
                    x++;
                    //test += mask[(halfRowNum - y) *colNum  + x];
                }
                break;
                //while(x<boundX) {mask[/*对应位置*/] = 0; x++;}
            }
        }
        //printf("%d  - %f\n", y, test);
    }
}

__device__
void get_sum_furnitureMsk(unsigned char* mask, int colCount, int rowCount, float * res, int absThreadIdx, int threadStride){
    for(int row = absThreadIdx; row<rowCount; row+=threadStride){
        for(int col =0; col<colCount; col++){
            if(mask[row*colCount + col] > 0)
                *res+=1;
        }
    }
    //printf("%d - %f\n", threadIdx.x, *res);
}
__device__
void change_an_obj_mask(sharedRoom * room, singleObj * obj, unsigned char* mask,
                        float* tmpSlot, int absThreadIdx, int threadStride){
    update_mask_by_object(mask, tmpSlot, obj->lastVertices, obj->lastBoundingBox,
                          room->rowCount/2, room->colCount,
                          absThreadIdx, threadStride, -1);
    update_mask_by_object(mask, tmpSlot, obj->vertices, obj->boundingBox,
                        room->rowCount/2, room->colCount,
                        absThreadIdx, threadStride, 1);
}
__device__
void sumUpMask(sharedRoom * room, unsigned char* mask, float * tmpSlot, float*dest, int nThreads){
    tmpSlot[threadIdx.x] = 0;
    get_sum_furnitureMsk(mask, room->colCount, room->rowCount, &tmpSlot[threadIdx.x], threadIdx.x, nThreads);

    __syncthreads();

    sumUp_dataInShare(tmpSlot, dest, nThreads);
}
__device__
mRect2f get_circulate_boundingbox(sharedRoom * room, mRect2f* rect){
     mRect2f nrect;
     nrect.x = ((rect->x-PERSONSIZE)< -room->half_width)? -room->half_width:rect->x-PERSONSIZE;
     nrect.y = ((rect->y + PERSONSIZE)> room->half_height)?  room->half_height:rect->y + PERSONSIZE;
     nrect.width = rect->width+2*PERSONSIZE;
     nrect.height = rect->height + 2* PERSONSIZE;
     return nrect;
}
__device__
void change_an_obj_backupMask(sharedRoom* room, singleObj * obj, unsigned char* mask, int nThreads){
    mRect2f rect = get_circulate_boundingbox(room, &obj->lastBoundingBox);

    update_mask_by_boundingBox(mask, rect, room->rowCount/2, room->colCount, threadIdx.x, nThreads, -1);

    rect = get_circulate_boundingbox(room, &obj->boundingBox);

    update_mask_by_boundingBox(mask, rect, room->rowCount/2, room->colCount, threadIdx.x, nThreads, 1);

}

__device__
void storeOrigin(singleObj * obj){
    for(int i=0;i<8;i++)
        obj->lastVertices[i] = obj->vertices[i];
    for(int i=0;i<3;i++)
        obj->lastTransAndRot[i] = obj->translation[i];
    obj->lastTransAndRot[3] = obj->zrotation;
    obj->lastBoundingBox = obj->boundingBox;
}

__device__
void restoreOrigin(sharedRoom * room, unsigned char* mask, float * tmpSlot, singleObj * obj, int nThreads){
    update_mask_by_object(mask, tmpSlot, obj->vertices, obj->boundingBox,
                      room->rowCount/2, room->colCount,
                      threadIdx.x, nThreads, -1);

    update_mask_by_object(mask, tmpSlot, obj->lastVertices, obj->lastBoundingBox,
                        room->rowCount/2, room->colCount,
                        threadIdx.x, nThreads, 1);

    for(int i=0;i<8;i++)
        obj->vertices[i] = obj->lastVertices[i];
    for(int i=0;i<3;i++)
        obj->translation[i] = obj->lastTransAndRot[i];
    obj->zrotation = obj->lastTransAndRot[3];
    obj->boundingBox = obj->lastBoundingBox;

    __syncthreads();
}
