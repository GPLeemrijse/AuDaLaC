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

	__host__ size_t created_instances_offset(void);

	// WILL BE DEPRECATED
	__host__ __device__ inst_size nrof_instances(void);

	__host__ __device__ inst_size nrof_instances2(bool step_parity);

	__host__ __device__ inst_size difference(void);

	__device__ __inline__ void update_counters(bool parity) {
		executing_instances[(uint)parity].fetch_max(
			created_instances.load(cuda::memory_order_relaxed)
		);
	}

	__host__ __device__ __inline__ void set_active_to_created(void){
		active_instances.store(created_instances.load(cuda::memory_order_relaxed), cuda::memory_order_relaxed);
	}

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
	cuda::atomic<inst_size, cuda::thread_scope_device> executing_instances[2]; // How many are part of this and the next iteration?
	cuda::atomic<inst_size, cuda::thread_scope_device> created_instances; // How many have been created in total?

	__device__ __inline__ RefType claim_instance2(void) {
		ADL::RefType slot = created_instances.fetch_add(1, cuda::memory_order_relaxed);
		
		//if(!slot) {asm("trap;");}
		assert(slot < capacity); // Incurs a stacksize penalty.
		
		return slot;
	}

	bool is_initialised;

	uint nrof_parameters;
};

#endif