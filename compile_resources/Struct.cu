#include "Struct.h"
#include "ADL.h"
#include <assert.h>

__host__ Struct::Struct(void) : is_initialised(false) {}

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

__host__ __device__ inline ADL::RefType Struct::claim_instance(void) {
	#ifdef __CUDA_ARCH__
	    ADL::RefType slot = atomicInc(&instantiated_instances, capacity);
	#else
	    ADL::RefType slot = instantiated_instances++;
	#endif
	return slot;
}

__host__ __device__ bool Struct::is_active(RefType instance){
	return instance < active_instances;
}

__host__ void* Struct::to_device(void) {
	void* device_ptr;
	size_t s = child_size();

	CHECK(
		cudaMalloc(&device_ptr, s)
	);

	CHECK(
		cudaMemcpy(device_ptr, this, s, cudaMemcpyHostToDevice)
	);

	return device_ptr;
}

__host__ void Struct::initialise(InitFile::StructInfo* info, inst_size capacity){
	assertCorrectInfo(info);
	assert (info->nrof_instances < capacity);

	void** params = get_parameters();
	nrof_parameters = info->parameter_data.size();

	for (int p = 0; p < nrof_parameters; p++){
		size_t param_size = size_of_type(info->parameter_types[p]);

		CHECK(
			cudaMalloc(&params[p], param_size * capacity)
		);
		
		// Copy initial instances
		CHECK(
			cudaMemcpyAsync(
				&((uint8_t*)params[p])[param_size], // free first slot for null instance
				info->parameter_data[p],
				param_size * info->nrof_instances,
				cudaMemcpyHostToDevice
			)
		);

		// Copy null-instance
		CHECK(
			cudaMemcpyAsync(
				params[p],
				ADL::default_value(info->parameter_types[p]),
				param_size,
				cudaMemcpyHostToDevice
			)
		);
	}

	instantiated_instances = info->nrof_instances + 1; // null-instance
	active_instances = info->nrof_instances + 1;
	capacity = capacity;
	is_initialised = true;
}