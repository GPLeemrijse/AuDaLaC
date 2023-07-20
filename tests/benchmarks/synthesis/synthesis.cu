#define ATOMIC(T) cuda::atomic<T, cuda::thread_scope_device>
#define STORE(A, V) A.store(V, cuda::memory_order_relaxed)
#define LOAD(A) A.load(cuda::memory_order_relaxed)

#define WLOAD(T, A) *((T*)&A)
#define ACQLOAD(A) A.load(cuda::memory_order_acquire)
#define WSTORE(T, A, V) *((T*)&A) = V
#define RELSTORE(A, V) A.store(V, cuda::memory_order_release)

#define FP_DEPTH 2

#include "ADL.h"
#include "Schedule.h"
#include "Struct.h"
#include "init_file.h"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
#include <tuple>
#include <vector>


class State : public Struct {
public:
	State (void) : Struct() {}
	
	ATOMIC(BoolType)* marked;
	ATOMIC(BoolType)* initial;
	ATOMIC(BoolType)* deleted;
	ATOMIC(RefType)* m_route;
	ATOMIC(BoolType)* supervisor;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "State");
		assert (info->parameter_types.size() == 5);
		assert (info->parameter_types[0] == ADL::Bool);
		assert (info->parameter_types[1] == ADL::Bool);
		assert (info->parameter_types[2] == ADL::Bool);
		assert (info->parameter_types[3] == ADL::Ref);
		assert (info->parameter_types[4] == ADL::Bool);
	};

	void** get_parameters(void) {
		return (void**)&marked;
	}

	size_t child_size(void) {
		return sizeof(State);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[5] = {
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(BoolType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(BoolType _marked,
									   BoolType _initial,
									   BoolType _deleted,
									   RefType _m_route,
									   BoolType _supervisor,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(marked[slot], _marked);
		STORE(initial[slot], _initial);
		STORE(deleted[slot], _deleted);
		STORE(m_route[slot], _m_route);
		STORE(supervisor[slot], _supervisor);
		*stable = false;
		return slot;
	}
};

class ControllableEvent : public Struct {
public:
	ControllableEvent (void) : Struct() {}
	
	ATOMIC(RefType)* x;
	ATOMIC(RefType)* y;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "ControllableEvent");
		assert (info->parameter_types.size() == 2);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&x;
	}

	size_t child_size(void) {
		return sizeof(ControllableEvent);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[2] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _x,
									   RefType _y,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(x[slot], _x);
		STORE(y[slot], _y);
		*stable = false;
		return slot;
	}
};

class UncontrollableEvent : public Struct {
public:
	UncontrollableEvent (void) : Struct() {}
	
	ATOMIC(RefType)* x;
	ATOMIC(RefType)* y;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "UncontrollableEvent");
		assert (info->parameter_types.size() == 2);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&x;
	}

	size_t child_size(void) {
		return sizeof(UncontrollableEvent);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[2] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _x,
									   RefType _y,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(x[slot], _x);
		STORE(y[slot], _y);
		*stable = false;
		return slot;
	}
};

using namespace cooperative_groups;


ControllableEvent host_ControllableEvent = ControllableEvent();
State host_State = State();
UncontrollableEvent host_UncontrollableEvent = UncontrollableEvent();

ControllableEvent* host_ControllableEvent_ptr = &host_ControllableEvent;
State* host_State_ptr = &host_State;
UncontrollableEvent* host_UncontrollableEvent_ptr = &host_UncontrollableEvent;

__device__ ControllableEvent* __restrict__ controllableevent;
__device__ State* __restrict__ state;
__device__ UncontrollableEvent* __restrict__ uncontrollableevent;

__device__ uint nrof_supervisor = 0;
__device__ uint nrof_deleted = 0;
#define FP_DEPTH 2
__device__ volatile bool fp_stack[FP_DEPTH];

__device__ void clear_stack(int lvl){
	while(lvl >= 0){
		fp_stack[lvl--] = false;
	}
}


typedef void(*step_func)(RefType, bool*);
template <step_func Step>
__device__ void executeStep(inst_size nrof_instances, grid_group grid, thread_block block, bool* stable){
	for(RefType self = grid.thread_rank(); self < nrof_instances; self += grid.size()){

		Step(self, stable);
	}
}

__host__ std::tuple<dim3, dim3> get_launch_dims(inst_size max_nrof_executing_instances, const void* kernel, bool print = false){
	int min_grid_size;
	int dyn_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &dyn_block_size, kernel, 0, 0);

	int numBlocksPerSm = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, dyn_block_size, 0);
  
	int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
	int wanted_blocks = (max_nrof_executing_instances + dyn_block_size - 1)/dyn_block_size;
	int used_blocks = min(max_blocks, wanted_blocks);
	int nrof_threads = used_blocks * dyn_block_size;

	if (used_blocks == 0) {
		fprintf(stderr, "Could not fit kernel on device!\n");
		exit(1234);
	}

	if (print) {
		fprintf(stderr, "A maximum of %u instances will execute.\n", max_nrof_executing_instances);
		fprintf(stderr, "Launching %u/%u blocks of %u threads = %u threads.\n", used_blocks, max_blocks, dyn_block_size, nrof_threads);
		fprintf(stderr, "Resulting in max %u instances per thread.\n", (max_nrof_executing_instances + nrof_threads - 1) / nrof_threads);
	}

	dim3 dimBlock(dyn_block_size, 1, 1);
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
__device__ void WeakSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {
	if (owner != 0){
		T old_val = WLOAD(T, params[owner]);
		if (old_val != new_val){
			WSTORE(T, params[owner], new_val);
			*stable = false;
		}
	}
}

template<typename T>
__device__ void RelSetParam(const RefType owner, ATOMIC(T) * const params, const T new_val, bool* stable) {
	if (owner != 0){
		T old_val = ACQLOAD(params[owner]);
		if (old_val != new_val){
			RELSTORE(params[owner], new_val);
			*stable = false;
		}
	}
}

__device__ void execute_State_init_supervisor(RefType self,
											  bool* stable){
	
	// supervisor := initial && !deleted;
	WeakSetParam<BoolType>(self, state->supervisor, (WLOAD(BoolType, state->initial[self]) && (!WLOAD(BoolType, state->deleted[self]))), stable);
}

__device__ void execute_State_count_supervisor(RefType self,
											   bool* stable){
	
	if (((self != 0) && WLOAD(BoolType, state->supervisor[self]))) {
		atomicInc(&nrof_supervisor, 0xffffffff);
	}
}

__device__ void execute_State_print_supervisor(RefType self,
											   bool* stable){
	
	if ((self == 0)) {
		printf("Nrof supervisor states: %u\n", nrof_supervisor);
	}
}

__device__ void execute_State_count_deleted(RefType self,
											bool* stable){
	
	if (((self != 0) && WLOAD(BoolType, state->deleted[self]))) {
		atomicInc(&nrof_deleted, 0xffffffff);
	}
}

__device__ void execute_State_print_deleted(RefType self,
											bool* stable){
	
	if ((self == 0)) {
		printf("Nrof deleted states: %u\n", nrof_deleted);
	}
}

__device__ void execute_State_delete_if_cant_reach_marked(RefType self,
														  bool* stable){
	
	if (((WLOAD(RefType, state->m_route[self]) == 0) && (!WLOAD(BoolType, state->marked[self])))) {
		// deleted := true;
		WeakSetParam<BoolType>(self, state->deleted, true, stable);
	}
}

__device__ void execute_State_collapse_m_routes(RefType self,
												bool* stable){
	
	if (((WLOAD(RefType, state->m_route[self]) != 0) && (WLOAD(BoolType, state->deleted[WLOAD(RefType, controllableevent->y[WLOAD(RefType, state->m_route[self])])]) || ((WLOAD(RefType, state->m_route[WLOAD(RefType, controllableevent->y[WLOAD(RefType, state->m_route[self])])]) == 0) && (!WLOAD(BoolType, state->marked[WLOAD(RefType, controllableevent->y[WLOAD(RefType, state->m_route[self])])])))))) {
		// m_route := null;
		WeakSetParam<RefType>(self, state->m_route, 0, stable);
	}
}

__device__ void execute_State_print(RefType self,
									bool* stable){
		if (self != 0) {
		printf("State(%u): marked=%u, initial=%u, deleted=%u, m_route=%u, supervisor=%u\n", self, LOAD(state->marked[self]), LOAD(state->initial[self]), LOAD(state->deleted[self]), LOAD(state->m_route[self]), LOAD(state->supervisor[self]));
	}

}

__device__ void execute_ControllableEvent_expand_m_route(RefType self,
														 bool* stable){
	
	if (((!WLOAD(BoolType, state->deleted[WLOAD(RefType, controllableevent->x[self])])) && (!WLOAD(BoolType, state->deleted[WLOAD(RefType, controllableevent->y[self])])))) {
		if (((LOAD(state->m_route[WLOAD(RefType, controllableevent->x[self])]) == 0) && ((LOAD(state->m_route[WLOAD(RefType, controllableevent->y[self])]) != 0) || WLOAD(BoolType, state->marked[WLOAD(RefType, controllableevent->y[self])])))) {
			// x.m_route := this;
			SetParam<RefType>(WLOAD(RefType, controllableevent->x[self]), state->m_route, self, stable);
		}
	}
}

__device__ void execute_ControllableEvent_expand_supervisor(RefType self,
															bool* stable){
	
	if (((!WLOAD(BoolType, state->deleted[WLOAD(RefType, controllableevent->x[self])])) && (!WLOAD(BoolType, state->deleted[WLOAD(RefType, controllableevent->y[self])])))) {
		if (LOAD(state->supervisor[WLOAD(RefType, controllableevent->x[self])])) {
			// y.supervisor := true;
			SetParam<BoolType>(WLOAD(RefType, controllableevent->y[self]), state->supervisor, true, stable);
		}
	}
}

__device__ void execute_ControllableEvent_print(RefType self,
												bool* stable){
		if (self != 0) {
		printf("ControllableEvent(%u): x=%u, y=%u\n", self, LOAD(controllableevent->x[self]), LOAD(controllableevent->y[self]));
	}

}

__device__ void execute_UncontrollableEvent_delete_if_del_state_is_reachable_via_uncontrollable(RefType self,
																								bool* stable){
	
	if (LOAD(state->deleted[WLOAD(RefType, uncontrollableevent->y[self])])) {
		// x.deleted := true;
		SetParam<BoolType>(WLOAD(RefType, uncontrollableevent->x[self]), state->deleted, true, stable);
	}
}

__device__ void execute_UncontrollableEvent_print(RefType self,
												  bool* stable){
		if (self != 0) {
		printf("UncontrollableEvent(%u): x=%u, y=%u\n", self, LOAD(uncontrollableevent->x[self]), LOAD(uncontrollableevent->y[self]));
	}

}


__global__ void kernel_State_init_supervisor(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_init_supervisor(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_count_supervisor(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_count_supervisor(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_print_supervisor(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_print_supervisor(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_count_deleted(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_count_deleted(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_print_deleted(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_print_deleted(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_delete_if_cant_reach_marked(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_delete_if_cant_reach_marked(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_collapse_m_routes(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_collapse_m_routes(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_State_print(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = state->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_State_print(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_ControllableEvent_expand_m_route(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = controllableevent->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_ControllableEvent_expand_m_route(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_ControllableEvent_expand_supervisor(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = controllableevent->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_ControllableEvent_expand_supervisor(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_ControllableEvent_print(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = controllableevent->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_ControllableEvent_print(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_UncontrollableEvent_delete_if_del_state_is_reachable_via_uncontrollable(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = uncontrollableevent->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_UncontrollableEvent_delete_if_del_state_is_reachable_via_uncontrollable(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void kernel_UncontrollableEvent_print(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = uncontrollableevent->nrof_instances();
		const uint bl_rank = this_thread_block().thread_rank();
	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}
	bool* stable_ptr = (bool*)&stable[bl_rank % 32];
	if(fp_lvl >= 0) __syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_UncontrollableEvent_print(self, stable_ptr);
	}
		if(fp_lvl >= 0){
		bool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
		if(!stable_reduced){
			clear_stack(fp_lvl);
		}
	}

}

__global__ void relaunch_fp_kernel(int fp_lvl,
								   cudaGraphExec_t restart,
								   cudaGraphExec_t exit){
	if(!fp_stack[fp_lvl]){
		fp_stack[fp_lvl] = true;
		CHECK(cudaGraphLaunch(restart, cudaStreamGraphTailLaunch));
	}
	else if (exit != NULL){
		CHECK(cudaGraphLaunch(exit, cudaStreamGraphTailLaunch));
	}
}

__global__ void launch_kernel(cudaGraphExec_t graph){
	CHECK(cudaGraphLaunch(graph, cudaStreamGraphTailLaunch));
}



int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}

	std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1048576);
	CHECK(cudaHostRegister(&host_ControllableEvent, sizeof(ControllableEvent), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_State, sizeof(State), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_UncontrollableEvent, sizeof(UncontrollableEvent), cudaHostRegisterDefault));

	inst_size ControllableEvent_capacity = structs[0].nrof_instances + 1;
	host_ControllableEvent.initialise(&structs[0], ControllableEvent_capacity);
	inst_size State_capacity = structs[1].nrof_instances + 1;
	host_State.initialise(&structs[1], State_capacity);
	inst_size UncontrollableEvent_capacity = structs[2].nrof_instances + 1;
	host_UncontrollableEvent.initialise(&structs[2], UncontrollableEvent_capacity);

	inst_size max_nrof_executing_instances = max(UncontrollableEvent_capacity, max(State_capacity, ControllableEvent_capacity));
	CHECK(cudaDeviceSynchronize());

	ControllableEvent * const loc_controllableevent = (ControllableEvent*)host_ControllableEvent.to_device();
	State * const loc_state = (State*)host_State.to_device();
	UncontrollableEvent * const loc_uncontrollableevent = (UncontrollableEvent*)host_UncontrollableEvent.to_device();

	CHECK(cudaMemcpyToSymbol(controllableevent, &loc_controllableevent, sizeof(ControllableEvent * const)));
	CHECK(cudaMemcpyToSymbol(state, &loc_state, sizeof(State * const)));
	CHECK(cudaMemcpyToSymbol(uncontrollableevent, &loc_uncontrollableevent, sizeof(UncontrollableEvent * const)));

	cudaStream_t kernel_stream;
	CHECK(cudaStreamCreate(&kernel_stream));
	Schedule schedule((void*)launch_kernel, (void*)relaunch_fp_kernel);

	schedule.begin_fixpoint();
		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_ControllableEvent_expand_m_route, ControllableEvent_capacity, 128);
		schedule.end_fixpoint();

		schedule.add_step((void*)kernel_State_delete_if_cant_reach_marked, State_capacity, 128);
		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_UncontrollableEvent_delete_if_del_state_is_reachable_via_uncontrollable, UncontrollableEvent_capacity, 128);
		schedule.end_fixpoint();

		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_State_collapse_m_routes, State_capacity, 128);
		schedule.end_fixpoint();

	schedule.end_fixpoint();

	schedule.add_step((void*)kernel_State_init_supervisor, State_capacity, 128);
	schedule.begin_fixpoint();
		schedule.add_step((void*)kernel_ControllableEvent_expand_supervisor, ControllableEvent_capacity, 128);
	schedule.end_fixpoint();

	schedule.add_step((void*)kernel_State_count_supervisor, State_capacity, 128);
	schedule.add_step((void*)kernel_State_print_supervisor, State_capacity, 128);	cudaGraphExec_t graph_exec = schedule.instantiate(kernel_stream);
//	schedule.print_dot();
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, kernel_stream);


	CHECK(cudaGraphLaunch(graph_exec, kernel_stream));

	cudaEventRecord(stop, kernel_stream);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Total walltime GPU: %0.6f ms\n", ms);

}
