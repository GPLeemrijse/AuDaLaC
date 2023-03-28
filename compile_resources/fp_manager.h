#ifndef FP_MANAGER_H
#define FP_MANAGER_H

#include "ADL.h"
#include <assert.h>

using namespace ADL;

class FPManager {
public:
	__host__ __device__ FPManager(uint depth) : device(false), depth(depth), current(0) {
		assert (depth < MAX_FP_DEPTH);
	}

	__host__ FPManager* to_device(void) {
		FPManager* device_ptr;

		// Allocate space for FPManager object
		CHECK(cudaMalloc(&device_ptr, sizeof(FPManager)));

		CHECK(cudaMemcpy(device_ptr, this, sizeof(FPManager), cudaMemcpyHostToDevice));

		bool _true = true;
		CHECK(cudaMemcpy(&device_ptr->device, &_true, sizeof(bool), cudaMemcpyHostToDevice));
		
		return device_ptr;
	}

	__host__ __device__ inline void push(void) {
		current++;
	}

	__host__ __device__ inline void pop(void) {
		current--;
	}

	__host__ __device__ inline void set(void) {
		stack[current-1] = false;
	}

	__host__ __device__ inline void reset(void) {
		stack[current-1] = true;
	}

	__host__ __device__ inline void clear(void) {
		for (int i = 0; i < current; i++) {
			stack[i] = false;
		}
	}

	__host__ __device__ inline bool done(void) {
		return stack[current-1] == true;
	}

	// other gets values of this
	__host__ void copy_to(FPManager* other) {
		// Copy stack + current
		CHECK(
			cudaMemcpy(
				&other->stack, // dst
				&this->stack,
		    	sizeof(bool) * MAX_FP_DEPTH + sizeof(uint),
		    	cudaMemcpyHostToDevice
			)
		);
	}

	// this gets values of other
	__host__ void copy_from(FPManager* other) {
		// Copy stack + current
		CHECK(
			cudaMemcpy(
				&this->stack, // dst
				&other->stack,
		    	sizeof(bool) * MAX_FP_DEPTH + sizeof(uint),
		    	cudaMemcpyDeviceToHost
			)
		);
	}

private:
	bool stack[MAX_FP_DEPTH];
	uint current;
	bool device;
	uint depth;
};

#endif