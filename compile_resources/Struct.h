#ifndef STRUCT_H
#define STRUCT_H

#include "ADL.h"
#include "init_file.h"

using namespace ADL;

class Struct {
public:
	__host__ explicit Struct(void);

	__host__ void free(void);

	__host__ void initialise(InitFile::StructInfo* info, inst_size capacity);

	virtual void** get_parameters(void) = 0;

	virtual void assertCorrectInfo(InitFile::StructInfo* info) = 0;

	__host__ __device__ bool is_active(RefType instance);

	__host__ void* to_device(void);

private:
	virtual size_t child_size(void) = 0;

	inst_size active_instances; // How many are part of the current iteration?
	inst_size instantiated_instances; // How many have been created in total?

	inst_size capacity; // For how many is space allocated?

	__host__ __device__ inline RefType claim_instance(void);

	bool is_initialised;

	uint nrof_parameters;
};

#endif