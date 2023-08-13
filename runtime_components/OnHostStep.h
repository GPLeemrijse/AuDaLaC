#ifndef ON_HOST_STEP_H
#define ON_HOST_STEP_H

#include "ADL.h"
using namespace ADL;

class OnHostStep {
	void* kernel;
	void* args[3];
	dim3 gridDim;
	dim3 blockDim;
	const char* name;
	inst_size prev_nrof_instances;

	public:
	OnHostStep(const char* name, void* kernel, inst_size* nrof_instances, int* fp_lvl, void* iter_idx);

	void launch(cudaStream_t stream);

	private:
	void update_dims(void);
};

#endif