#pragma once
#include <curand.h>
#include <curand_kernel.h>
__device__
float dist_between_points(const float* pos1, const float* pos2) {
    return sqrtf(powf((pos1[0] - pos2[0]),2.0f) + powf((pos1[1] - pos2[1]),2.0f) +powf((pos1[2] - pos2[2]),2.0f));
}
__device__
float t(float d, float m, float M, int a = 2){
    if (d < m)
		return powf((d / m), float(a));
	else if (d > M)
		return powf((M / d), float(a));
	else
		return 1.0f;
}
__device__
void rot_around_a_point(float center[3], float * x, float * y, float s, float c) {
	// translate point back to origin:
	*x -= center[0];
	*y -= center[1];

	// rotate point
	float xnew = *x * c - *y * s;
	float ynew = *x * s + *y * c;

	// translate point back:
	*x = xnew + center[0];
	*y = ynew + center[1];
}

__device__
float sumUp_weighted_dataInShare(float* data, float* weights, int bound){
    float res = 0;
    for(int i=0; i<bound; i++)
        res += fabsf(data[i])* weights[i];
    return res;
}
__device__
void sumUp_dataInShare(float * data, float* res, int bound){
    int i = (*res ==0)?0:1;
    for(; i<bound; i++)
        *res += data[i];
}

__device__
float density_function(float beta, float cost) {
    //printf("%f\n", beta*cost);
	return exp2f(-beta * cost);
}

__device__
void get_random_state(curandState_t * state, int index){
    //seed, sequence number(multiple cores), offset
    curand_init((unsigned long long )clock() + index, 0, 0, state);
}
__device__
float get_normal_random(float mean, float devision,int index = blockIdx.x * blockDim.x + threadIdx.x){
    curandState_t state;
    get_random_state(&state, index);
    return curand_normal(&state) * devision/2 + mean;
}
__device__
int get_int_random(int maxLimit,int index = blockIdx.x * blockDim.x + threadIdx.x){
    curandState_t state;
    get_random_state(&state, index);
    return curand(&state)%maxLimit;
}
__device__
float get_float_random(int maxLimit, int index = blockIdx.x * blockDim.x + threadIdx.x) {
    curandState_t state;
    get_random_state(&state, index);
    return curand_uniform(&state) * maxLimit;
}
__device__
bool point_in_rectangle(float* tmpSlot, const float *vertices, float x, float y){
    int nCross = 0;
    	tmpSlot[2] = vertices[0]; tmpSlot[3] = vertices[1];
    	for (int i = 1; i <= 4; i++) {
    		tmpSlot[0] = tmpSlot[2]; tmpSlot[1] = tmpSlot[3];
    		if (i == 4) {
    			tmpSlot[2] = vertices[0]; tmpSlot[3] = vertices[1];
    		}
    		else {
    			tmpSlot[2] = vertices[2 * i]; tmpSlot[3] = vertices[2 * i + 1];
    		}
    		if (tmpSlot[1] == tmpSlot[3])
    			continue;
    		if (y < tmpSlot[1] && y < tmpSlot[3])
    			continue;
    		if (y > tmpSlot[1] && y > tmpSlot[3])
    			continue;
    		float tx = (y - tmpSlot[1])*(tmpSlot[2] - tmpSlot[0]) / (tmpSlot[3] - tmpSlot[1]) + tmpSlot[0];
    		if (tx >= x)
    			nCross++;
    	}
    	return (nCross % 2 == 1);
}
__device__
int binary_search_Inside_Point(int xl, int xr, int mid, int y, float* tmpSlot, const float* vertices){
    while (xl < xr) {
    mid = (xl + xr) / 2;
    if (point_in_rectangle(tmpSlot, vertices, mid, y))
        xl = mid + 1;
    else
        xr = mid - 1;
    }
    return xl;
}
