#define THREADS_PER_BLOCK 256
#define ATOMIC(T) cuda::atomic<T, cuda::thread_scope_device>
#define STORE(A, V) A.store(V, cuda::memory_order_relaxed)
#define LOAD(A) A.load(cuda::memory_order_relaxed)

#define WLOAD(T, A) *((T*)&A)
#define WSTORE(T, A, V) *((T*)&A) = V

#define ListElem_MASK (((uint16_t)1) << 0)
#define STEP_PARITY(STRUCT) ((bool)(struct_step_parity & STRUCT ## _MASK))
#define TOGGLE_STEP_PARITY(STRUCT) {struct_step_parity ^= STRUCT ## _MASK;}


#include "ADL.h"
#include "Struct.h"
#include "init_file.h"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
#include <tuple>
#include <vector>


class ListElem : public Struct {
public:
	ListElem (void) : Struct() {}
	
	ATOMIC(IntType)* val;
	ATOMIC(RefType)* prev;
	ATOMIC(RefType)* auxprev;
	ATOMIC(IntType)* auxval;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "ListElem");
		assert (info->parameter_types.size() == 4);
		assert (info->parameter_types[0] == ADL::Int);
		assert (info->parameter_types[1] == ADL::Ref);
		assert (info->parameter_types[2] == ADL::Ref);
		assert (info->parameter_types[3] == ADL::Int);
	};

	void** get_parameters(void) {
		return (void**)&val;
	}

	size_t child_size(void) {
		return sizeof(ListElem);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[4] = {
			sizeof(ATOMIC(IntType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(IntType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(IntType _val,
									   RefType _prev,
									   RefType _auxprev,
									   IntType _auxval,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(val[slot], _val);
		STORE(prev[slot], _prev);
		STORE(auxprev[slot], _auxprev);
		STORE(auxval[slot], _auxval);
		*stable = false;
		return slot;
	}
};

using namespace cooperative_groups;

ListElem host_ListElem = ListElem();

ListElem* host_ListElem_ptr = &host_ListElem;

__device__ ListElem* __restrict__ listelem;


#define FP_DEPTH 1
/* Transform an iter_idx into the fp_stack index
   associated with that operation.
*/
#define FP_SET(X) (X)
#define FP_RESET(X) ((X) + 1 >= 3 ? (X) + 1 - 3 : (X) + 1)
#define FP_READ(X) ((X) + 2 >= 3 ? (X) + 2 - 3 : (X) + 2)

__device__ cuda::atomic<bool, cuda::thread_scope_device> fp_stack[FP_DEPTH][3];

__device__ void clear_stack(int lvl, uint8_t* iter_idx) {
	/*	Clears the stack on the FP_SET side.
		The FP_RESET and FP_READ sides should remain the same.
	*/
	while(lvl >= 0){
		fp_stack[lvl][FP_SET(iter_idx[lvl])].store(false, cuda::memory_order_relaxed);
		lvl--;
	}
}

typedef void(*step_func)(RefType, bool*);
template <step_func Step>
__device__ void executeStep(inst_size nrof_instances, grid_group grid, thread_block block, bool* stable){
	for(RefType self = grid.thread_rank(); self < nrof_instances; self += grid.size()){

		Step(self, stable);
    }
}

__host__ std::tuple<dim3, dim3> get_launch_dims(inst_size max_nrof_executing_instances, const void* kernel){
  int numBlocksPerSm = 0;
  int tpb = THREADS_PER_BLOCK;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, tpb, 0);
  
  int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
  int wanted_blocks = (max_nrof_executing_instances + tpb - 1)/tpb;
  int used_blocks = min(max_blocks, wanted_blocks);

  fprintf(stderr, "Launching %u/%u blocks of %u threads = %u threads.\nResulting in max %u instances per thread.\n", used_blocks, max_blocks, tpb, used_blocks * tpb, (max_nrof_executing_instances + (used_blocks * tpb) - 1) / (used_blocks * tpb));

  dim3 dimBlock(tpb, 1, 1);
  dim3 dimGrid(used_blocks, 1, 1);
  return std::make_tuple(dimGrid, dimBlock);
}

template<typename T>
__device__ void SetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {
	if (owner != 0){
		T old_val = LOAD(params[owner]);
		if (old_val != new_val){
			STORE(params[owner], new_val);
			*stable = false;
		}
	}
}

template<typename T>
__device__ void WSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {
    if (owner != 0){
        T old_val = WLOAD(T, params[owner]);
        if (old_val != new_val){
            WSTORE(T, params[owner], new_val);
            *stable = false;
        }
    }
}

__device__ void ListElem_print(const RefType self,
							   bool* stable){
	if (self != 0) {
		printf("ListElem(%u): val=%d, prev=%u, auxprev=%u, auxval=%d\n", self, LOAD(listelem->val[self]), LOAD(listelem->prev[self]), LOAD(listelem->auxprev[self]), LOAD(listelem->auxval[self]));
	}
}

__device__ void ListElem_prefixOne(const RefType self,
								   bool* stable){
	
	// auxval := prev.val;
	WSetParam<IntType>(self, listelem->auxval, WLOAD(IntType, listelem->val[WLOAD(RefType, listelem->prev[self])]), stable);
	// auxprev := prev.prev;
	WSetParam<RefType>(self, listelem->auxprev, WLOAD(RefType, listelem->prev[WLOAD(RefType, listelem->prev[self])]), stable);
}

__device__ void ListElem_prefixTwo(const RefType self,
								   bool* stable){
	
	// val := val + auxval;
	WSetParam<IntType>(self, listelem->val, (WLOAD(IntType, listelem->val[self]) + WLOAD(IntType, listelem->auxval[self])), stable);
	// prev := auxprev;
	WSetParam<RefType>(self, listelem->prev, WLOAD(RefType, listelem->auxprev[self]), stable);
}

__device__ void ListElem_print_sol(const RefType self,
								   bool* stable){
	
	if ((self != 0)) {
		IntType ld_val = WLOAD(IntType, listelem->val[self]);
		printf("(%u, %d)\n", self, ld_val);
	}
}


__global__ void schedule_kernel(){
	const grid_group grid = this_grid();
	const thread_block block = this_thread_block();
	uint16_t struct_step_parity = 0; // bitmask
	bool stable = true; // Only used to compile steps outside fixpoints
	uint8_t iter_idx[FP_DEPTH] = {0}; // Denotes which fp_stack index ([0, 2]) is currently being set.

	do{
		bool stable = true;
		if (grid.thread_rank() == 0){
			/* Resets the next fp_stack index in advance. */
			fp_stack[0][FP_RESET(iter_idx[0])].store(true, cuda::memory_order_relaxed);
		}


		TOGGLE_STEP_PARITY(ListElem);
		executeStep<ListElem_prefixOne>(listelem->nrof_instances2(STEP_PARITY(ListElem)), grid, block, &stable);
		listelem->update_counters(!STEP_PARITY(ListElem));

		grid.sync();

		TOGGLE_STEP_PARITY(ListElem);
		executeStep<ListElem_prefixTwo>(listelem->nrof_instances2(STEP_PARITY(ListElem)), grid, block, &stable);
		listelem->update_counters(!STEP_PARITY(ListElem));
		if(!stable){
			clear_stack(0, iter_idx);
		}
		/* The next index to set is the one that has been reset. */
		iter_idx[0] = FP_RESET(iter_idx[0]);
		grid.sync();
	} while(!fp_stack[0][FP_READ(iter_idx[0])].load(cuda::memory_order_relaxed));


	TOGGLE_STEP_PARITY(ListElem);
	executeStep<ListElem_print_sol>(listelem->nrof_instances2(STEP_PARITY(ListElem)), grid, block, &stable);
	listelem->update_counters(!STEP_PARITY(ListElem));
}


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}

	std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576);
	CHECK(cudaHostRegister(&host_ListElem, sizeof(ListElem), cudaHostRegisterDefault));

	host_ListElem.initialise(&structs[0], structs[0].nrof_instances + 1);

	inst_size max_nrof_executing_instances = structs[0].nrof_instances + 1;
	CHECK(cudaDeviceSynchronize());

	ListElem * const loc_listelem = (ListElem*)host_ListElem.to_device();

	CHECK(cudaMemcpyToSymbol(listelem, &loc_listelem, sizeof(ListElem * const)));

	cuda::atomic<bool, cuda::thread_scope_device>* fp_stack_address;
	CHECK(cudaGetSymbolAddress((void **)&fp_stack_address, fp_stack));
	CHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 3 * sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));
	void* schedule_kernel_args[] = {};
	auto dims = get_launch_dims(max_nrof_executing_instances, (void*)schedule_kernel);


	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)schedule_kernel,
			std::get<0>(dims),
			std::get<1>(dims),
			schedule_kernel_args
		)
	);


	CHECK(cudaDeviceSynchronize());
}
