#include "Struct.h"
#include "ADL.h"
#include <assert.h>
#include <cuda/atomic>

__host__ Struct::Struct(void) : is_initialised(false), active_instances(0), instantiated_instances(0), capacity(0), nrof_parameters(0) {}

__host__ void Struct::free(void) {
	if (is_initialised) {
		void** params = get_parameters();

		for (int p = 0; p < nrof_parameters; p++){
			CHECK(
				cudaFree(params[p])
			);
		}
	}
}

__host__ size_t Struct::created_instances_offset(void) {
	return ((size_t)(&this->created_instances) - (size_t)this);
}

__host__ void* Struct::to_device(void* allocated_ptr) {
	void* device_ptr;
	size_t s = child_size();

	if (allocated_ptr != NULL) {
		device_ptr = allocated_ptr;
	} else {
		CHECK(
			cudaMalloc(&device_ptr, s)
		);
	}

	CHECK(
		cudaMemcpy(device_ptr, this, s, cudaMemcpyHostToDevice)
	);

	return device_ptr;
}

__host__ void Struct::initialise(InitFile::StructInfo* info, inst_size capacity){
	assert_correct_info(info);
	if (info->nrof_instances >= capacity) {
		fprintf(stderr, "Error: %u instances supplied while the capacity is %u.\n", info->nrof_instances, capacity);
	}
	assert (info->nrof_instances < capacity);

	void** params = get_parameters();
	this->nrof_parameters = info->parameter_data.size();

	for (int p = 0; p < this->nrof_parameters; p++){
		size_t info_param_size = size_of_type(info->parameter_types[p]);
		size_t actual_param_size = this->param_size(p);

		CHECK(
			cudaMalloc(&params[p], actual_param_size * capacity)
		);
		
		if (info_param_size == actual_param_size){
			// Copy initial instances
			CHECK(
				cudaMemcpyAsync(
					&((uint8_t*)params[p])[info_param_size], // free first slot for null instance
					info->parameter_data[p],
					info_param_size * info->nrof_instances,
					cudaMemcpyHostToDevice
				)
			);
		} else if (actual_param_size > info_param_size) {
			fprintf(stderr, "Strided copy for param %d of %s\n", p, info->name.c_str());
			for(int i = 0; i < info->nrof_instances; i++){
				// Copy initial instances
				CHECK(
					cudaMemcpyAsync(
						&((uint8_t*)params[p])[actual_param_size * i],
						&((uint8_t*)info->parameter_data[p])[info_param_size * i],
						info_param_size,
						cudaMemcpyHostToDevice
					)
				);
			}
		} else {
			throw std::bad_alloc();
		}

		// zero null-instance
		CHECK(
			cudaMemsetAsync(
				params[p],
				0,
				actual_param_size
			)
		);
	}

	this->instantiated_instances = info->nrof_instances + 1; // null-instance
	this->active_instances = info->nrof_instances + 1;
	this->created_instances = info->nrof_instances + 1;
	this->executing_instances[0].store(info->nrof_instances + 1, cuda::memory_order_relaxed);
	this->executing_instances[1].store(info->nrof_instances + 1, cuda::memory_order_relaxed);
	this->capacity = capacity;
	this->is_initialised = true;
}

__host__ __device__ inst_size Struct::nrof_instances(void){
	return active_instances.load(cuda::memory_order_seq_cst);
}

__host__ __device__ inst_size Struct::nrof_instances2(bool step_parity){
	return executing_instances[(uint)step_parity].load(cuda::memory_order_relaxed);
}

__host__ __device__ inst_size Struct::difference(void){
	return instantiated_instances.load(cuda::memory_order_seq_cst) - active_instances.load(cuda::memory_order_seq_cst);
}

/* Sets own active instances to all instantiated instances.
   Then copies own values to 'other'.
   Returns true iff there was a difference between active and instantiated instances.
*/
__host__ __device__ bool Struct::sync_nrof_instances(Struct* other) {
	// First sync own active instances
	inst_size instantiated_instances = this->instantiated_instances.load(cuda::memory_order_seq_cst);
	inst_size old_active_instances = this->active_instances.load(cuda::memory_order_seq_cst);
	bool differ = instantiated_instances != old_active_instances;

	if (differ) {
		this->active_instances.store(instantiated_instances, cuda::memory_order_seq_cst);
	}

	#ifdef __CUDA_ARCH__
		/*NOTE: Also copies to instantiated_instances, so keep those sequential in memory. */
		memcpy(
			&other->active_instances, // dst
			&this->active_instances,
			sizeof(cuda::atomic<inst_size, cuda::thread_scope_device>) * 2
		);
	#else
		/*NOTE: Also copies to instantiated_instances, so keep those sequential in memory. */
		CHECK(
			cudaMemcpy(
				&other->active_instances, // dst
				&this->active_instances,
				sizeof(cuda::atomic<inst_size, cuda::thread_scope_device>) * 2,
				cudaMemcpyHostToDevice
			)
		);
	#endif

	return differ;
}