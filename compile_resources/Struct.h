#ifndef STRUCT_H
#define STRUCT_H

#include "ADL.h"
#include "init_file.h"
#include <assert.h>

using namespace ADL;

class Struct {
public:
	__host__ explicit Struct(void);

	__host__ void free(void);

	__host__ void initialise(InitFile::StructInfo* info, inst_size capacity);

	virtual void** get_parameters(void) = 0;

	virtual void assert_correct_info(InitFile::StructInfo* info) = 0;

	__host__ __device__ bool is_active(RefType instance);

	__host__ void* to_device(void);

	// other gets values of this
	__host__ __device__ void sync_nrof_instances(Struct* other);

	__host__ __device__ inst_size nrof_instances(void);

protected:
	virtual size_t child_size(void) = 0;

	// Keep sequential in memory
	inst_size active_instances; // How many are part of the current iteration?
	inst_size instantiated_instances; // How many have been created in total?

	inst_size capacity; // For how many is space allocated?

	__host__ __device__ inline RefType claim_instance(void) {
		#ifdef __CUDA_ARCH__
		    ADL::RefType slot = atomicInc(&instantiated_instances, capacity);
		    assert(slot != 0);
		#else
		    ADL::RefType slot = instantiated_instances++;
		    assert(slot < capacity);
		#endif
		return slot;
	}

	bool is_initialised;

	uint nrof_parameters;
};

#endif