#ifndef STRUCT_H
#define STRUCT_H

#include "ADL.h"
#include "init_file.h"
#include <assert.h>
#include <cuda/atomic>

using namespace ADL;

class Struct {
public:
	__host__ explicit Struct(void);

	__host__ void free(void);

	__host__ void initialise(InitFile::StructInfo* info, inst_size capacity);

	virtual void** get_parameters(void) = 0;

	virtual void assert_correct_info(InitFile::StructInfo* info) = 0;

	__host__ void* to_device(void* allocated_ptr = NULL);

	// other gets values of this
	__host__ __device__ bool sync_nrof_instances(Struct* other);

	// WILL BE DEPRECATED
	__host__ __device__ inst_size nrof_instances(void);

	__host__ __device__ inst_size nrof_instances2(bool step_parity);

	__host__ __device__ inst_size difference(void);

protected:
	virtual size_t child_size(void) = 0;

	virtual size_t param_size(uint idx) = 0;

	// Keep sequential in memory  (WILL BE DEPRECATED)
	cuda::atomic<inst_size, cuda::thread_scope_device> active_instances; // How many are part of the current iteration?
	cuda::atomic<inst_size, cuda::thread_scope_device> instantiated_instances; // How many have been created in total?

	inst_size capacity; // For how many is space allocated?

	__host__ __device__ inline RefType claim_instance(void) {
		ADL::RefType slot = instantiated_instances.fetch_add(1, cuda::memory_order_relaxed);
		assert(slot < capacity);
		return slot;
	}

	// New one-less-sync method:
	inst_size executing_instances[2]; // How many are part of this and the next iteration?
	inst_size created_instances; // How many have been created in total?

	__device__ __inline__ RefType claim_instance2(bool step_parity) {
		ADL::RefType slot = atomicInc(&created_instances, capacity);
		
		// atomicInc wraps around to 0 if it exceeds capacity
		if(!slot) {asm("trap;");}
		//assert(slot); // Incurs a substantial stacksize penalty. 
		
		// Update the next iteration's executing_instances
		atomicMax(&executing_instances[(uint)!step_parity], slot);
		return slot;
	}

	bool is_initialised;

	uint nrof_parameters;
};

#endif