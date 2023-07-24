#include "OnHostStep.h"
#include "ADL.h"

OnHostStep::OnHostStep(const char* name, void* kernel, inst_size* nrof_instances, int* fp_lvl, void* iter_idx)
	: kernel(kernel), name(name) {
	args[0] = (void*)nrof_instances;
	args[1] = (void*)fp_lvl;
	args[2] = (void*)iter_idx;
	prev_nrof_instances = *nrof_instances;
	int min_grid_size;
	int dyn_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &dyn_block_size, kernel, 0, 0);

	blockDim = dim3(dyn_block_size, 1, 1);

	update_dims();
}

void OnHostStep::launch(cudaStream_t stream) {
	if (*((inst_size*)args[0]) != prev_nrof_instances){
		update_dims();
		prev_nrof_instances = *((inst_size*)args[0]);
	}

	CHECK(cudaLaunchKernel(kernel, gridDim, blockDim, args, 128, stream));
}

void OnHostStep::update_dims(void) {
	int used_blocks = (*((inst_size*)args[0]) + blockDim.x - 1) / blockDim.x;
	gridDim.x = used_blocks;
	//fprintf(stderr, \"Updated %s to %u/%u blocks of %u threads = %u threads.\\n\", name, used_blocks, max_blocks, blockDim.x, used_blocks * blockDim.x);
}