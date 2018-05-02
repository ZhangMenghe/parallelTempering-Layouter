__device__ __host__
void rot_around_point(float center[3], float * x, float * y, float s, float c) {
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
