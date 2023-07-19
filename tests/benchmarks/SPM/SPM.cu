#define THREADS_PER_BLOCK 256
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


class Node : public Struct {
public:
	Node (void) : Struct() {}
	
	ATOMIC(NatType)* p;
	ATOMIC(BoolType)* is_odd;
	ATOMIC(BoolType)* dirty;
	ATOMIC(BoolType)* rho_top;
	ATOMIC(NatType)* rho_1;
	ATOMIC(NatType)* rho_3;
	ATOMIC(RefType)* cand;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Node");
		assert (info->parameter_types.size() == 7);
		assert (info->parameter_types[0] == ADL::Nat);
		assert (info->parameter_types[1] == ADL::Bool);
		assert (info->parameter_types[2] == ADL::Bool);
		assert (info->parameter_types[3] == ADL::Bool);
		assert (info->parameter_types[4] == ADL::Nat);
		assert (info->parameter_types[5] == ADL::Nat);
		assert (info->parameter_types[6] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&p;
	}

	size_t child_size(void) {
		return sizeof(Node);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[7] = {
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(NatType _p,
									   BoolType _is_odd,
									   BoolType _dirty,
									   BoolType _rho_top,
									   NatType _rho_1,
									   NatType _rho_3,
									   RefType _cand,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(p[slot], _p);
		STORE(is_odd[slot], _is_odd);
		STORE(dirty[slot], _dirty);
		STORE(rho_top[slot], _rho_top);
		STORE(rho_1[slot], _rho_1);
		STORE(rho_3[slot], _rho_3);
		STORE(cand[slot], _cand);
		*stable = false;
		return slot;
	}
};

class Edge : public Struct {
public:
	Edge (void) : Struct() {}
	
	ATOMIC(RefType)* v;
	ATOMIC(RefType)* w;
	ATOMIC(NatType)* max1;
	ATOMIC(NatType)* max3;
	ATOMIC(BoolType)* prog_top;
	ATOMIC(NatType)* prog_1;
	ATOMIC(NatType)* prog_3;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Edge");
		assert (info->parameter_types.size() == 7);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
		assert (info->parameter_types[2] == ADL::Nat);
		assert (info->parameter_types[3] == ADL::Nat);
		assert (info->parameter_types[4] == ADL::Bool);
		assert (info->parameter_types[5] == ADL::Nat);
		assert (info->parameter_types[6] == ADL::Nat);
	};

	void** get_parameters(void) {
		return (void**)&v;
	}

	size_t child_size(void) {
		return sizeof(Edge);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[7] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(NatType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _v,
									   RefType _w,
									   NatType _max1,
									   NatType _max3,
									   BoolType _prog_top,
									   NatType _prog_1,
									   NatType _prog_3,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(v[slot], _v);
		STORE(w[slot], _w);
		STORE(max1[slot], _max1);
		STORE(max3[slot], _max3);
		STORE(prog_top[slot], _prog_top);
		STORE(prog_1[slot], _prog_1);
		STORE(prog_3[slot], _prog_3);
		*stable = false;
		return slot;
	}
};

using namespace cooperative_groups;


Edge host_Edge = Edge();
Node host_Node = Node();

Edge* host_Edge_ptr = &host_Edge;
Node* host_Node_ptr = &host_Node;

__device__ Edge* __restrict__ edge;
__device__ Node* __restrict__ node;

__device__ uint nrof_odd_wins = 0;
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
	int numBlocksPerSm = 0;
	int tpb = THREADS_PER_BLOCK;

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, tpb, 0);
  
	int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
	int wanted_blocks = (max_nrof_executing_instances + tpb - 1)/tpb;
	int used_blocks = min(max_blocks, wanted_blocks);
	int nrof_threads = used_blocks * tpb;

	if (used_blocks == 0) {
		fprintf(stderr, "Could not fit kernel on device!\n");
		exit(1234);
	}

	if (print) {
		fprintf(stderr, "A maximum of %u instances will execute.\n", max_nrof_executing_instances);
		fprintf(stderr, "Launching %u/%u blocks of %u threads = %u threads.\n", used_blocks, max_blocks, tpb, nrof_threads);
		fprintf(stderr, "Resulting in max %u instances per thread.\n", (max_nrof_executing_instances + nrof_threads - 1) / nrof_threads);
	}

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

__device__ void execute_Node_lift(RefType self,
								  bool* stable){
	
	// dirty := false;
	WeakSetParam<BoolType>(self, node->dirty, false, stable);
	if ((WLOAD(RefType, node->cand[self]) != 0)) {
		if (((((WLOAD(BoolType, edge->prog_top[WLOAD(RefType, node->cand[self])]) != WLOAD(BoolType, node->rho_top[self])) && WLOAD(BoolType, edge->prog_top[WLOAD(RefType, node->cand[self])])) || ((WLOAD(BoolType, edge->prog_top[WLOAD(RefType, node->cand[self])]) == WLOAD(BoolType, node->rho_top[self])) && (WLOAD(NatType, edge->prog_1[WLOAD(RefType, node->cand[self])]) > WLOAD(NatType, node->rho_1[self])))) || (((WLOAD(BoolType, edge->prog_top[WLOAD(RefType, node->cand[self])]) == WLOAD(BoolType, node->rho_top[self])) && (WLOAD(NatType, edge->prog_1[WLOAD(RefType, node->cand[self])]) == WLOAD(NatType, node->rho_1[self]))) && (WLOAD(NatType, edge->prog_3[WLOAD(RefType, node->cand[self])]) > WLOAD(NatType, node->rho_3[self]))))) {
			// rho_top := cand.prog_top;
			WeakSetParam<BoolType>(self, node->rho_top, WLOAD(BoolType, edge->prog_top[WLOAD(RefType, node->cand[self])]), stable);
			// rho_1 := cand.prog_1;
			WeakSetParam<NatType>(self, node->rho_1, WLOAD(NatType, edge->prog_1[WLOAD(RefType, node->cand[self])]), stable);
			// rho_3 := cand.prog_3;
			WeakSetParam<NatType>(self, node->rho_3, WLOAD(NatType, edge->prog_3[WLOAD(RefType, node->cand[self])]), stable);
			// dirty := true;
			WeakSetParam<BoolType>(self, node->dirty, true, stable);
		}
	}
}

__device__ void execute_Node_count_odd(RefType self,
									   bool* stable){
	
	if (((self != 0) && WLOAD(BoolType, node->rho_top[self]))) {
		atomicInc(&nrof_odd_wins, 0xffffffff);
	}
}

__device__ void execute_Node_print_odd(RefType self,
									   bool* stable){
	
	if ((self == 0)) {
		printf("Number of odd won vertices = %u\n", nrof_odd_wins);
	}
}

__device__ void execute_Node_print(RefType self,
								   bool* stable){
		if (self != 0) {
		printf("Node(%u): p=%u, is_odd=%u, dirty=%u, rho_top=%u, rho_1=%u, rho_3=%u, cand=%u\n", self, LOAD(node->p[self]), LOAD(node->is_odd[self]), LOAD(node->dirty[self]), LOAD(node->rho_top[self]), LOAD(node->rho_1[self]), LOAD(node->rho_3[self]), LOAD(node->cand[self]));
	}

}

__device__ void execute_Edge_prog(RefType self,
								  bool* stable){
	
	if (WLOAD(BoolType, node->dirty[WLOAD(RefType, edge->w[self])])) {
		BoolType m_top = WLOAD(BoolType, node->rho_top[WLOAD(RefType, edge->w[self])]);
		NatType m_1 = 0;
		if ((WLOAD(NatType, node->p[WLOAD(RefType, edge->v[self])]) >= 1)) {
			m_1 = WLOAD(NatType, node->rho_1[WLOAD(RefType, edge->w[self])]);
		}
		NatType m_3 = 0;
		if ((WLOAD(NatType, node->p[WLOAD(RefType, edge->v[self])]) >= 3)) {
			m_3 = WLOAD(NatType, node->rho_3[WLOAD(RefType, edge->w[self])]);
		}
		if (((WLOAD(NatType, node->p[WLOAD(RefType, edge->v[self])]) % 2) == 1)) {
			BoolType increased = m_top;
			if (((WLOAD(NatType, node->p[WLOAD(RefType, edge->v[self])]) >= 3) && (!increased))) {
				NatType new_3 = 0;
				if ((m_3 < WLOAD(NatType, edge->max3[self]))) {
					new_3 = (m_3 + 1);
					increased = true;
				}
				m_3 = new_3;
			}
			if (((WLOAD(NatType, node->p[WLOAD(RefType, edge->v[self])]) >= 1) && (!increased))) {
				NatType new_1 = 0;
				if ((m_1 < WLOAD(NatType, edge->max1[self]))) {
					new_1 = (m_1 + 1);
					increased = true;
				}
				m_1 = new_1;
			}
			if ((!increased)) {
				m_top = true;
			}
		}
		if ((LOAD(node->cand[WLOAD(RefType, edge->v[self])]) == self)) {
			BoolType changed = (((m_top != WLOAD(BoolType, edge->prog_top[self])) || (WLOAD(NatType, edge->prog_1[self]) != m_1)) || (WLOAD(NatType, edge->prog_3[self]) != m_3));
			if (changed) {
				// v.cand := null;
				SetParam<RefType>(WLOAD(RefType, edge->v[self]), node->cand, 0, stable);
			}
		}
		// prog_top := m_top;
		WeakSetParam<BoolType>(self, edge->prog_top, m_top, stable);
		// prog_1 := m_1;
		WeakSetParam<NatType>(self, edge->prog_1, m_1, stable);
		// prog_3 := m_3;
		WeakSetParam<NatType>(self, edge->prog_3, m_3, stable);
	}
}

__device__ void execute_Edge_minmax_top(RefType self,
										bool* stable){
	
	if (((LOAD(node->cand[WLOAD(RefType, edge->v[self])]) == 0) || ((((!WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])])) && WLOAD(BoolType, edge->prog_top[LOAD(node->cand[WLOAD(RefType, edge->v[self])])])) && (!WLOAD(BoolType, edge->prog_top[self]))) || ((WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])]) && (!WLOAD(BoolType, edge->prog_top[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]))) && WLOAD(BoolType, edge->prog_top[self]))))) {
		// v.cand := this;
		SetParam<RefType>(WLOAD(RefType, edge->v[self]), node->cand, self, stable);
	}
}

__device__ void execute_Edge_minmax_1(RefType self,
									  bool* stable){
	
	if (((WLOAD(BoolType, edge->prog_top[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) == WLOAD(BoolType, edge->prog_top[self])) && (((!WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])])) && (WLOAD(NatType, edge->prog_1[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) > WLOAD(NatType, edge->prog_1[self]))) || (WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])]) && (WLOAD(NatType, edge->prog_1[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) < WLOAD(NatType, edge->prog_1[self])))))) {
		// v.cand := this;
		SetParam<RefType>(WLOAD(RefType, edge->v[self]), node->cand, self, stable);
	}
}

__device__ void execute_Edge_minmax_3(RefType self,
									  bool* stable){
	
	if ((((WLOAD(BoolType, edge->prog_top[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) == WLOAD(BoolType, edge->prog_top[self])) && (WLOAD(NatType, edge->prog_1[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) == WLOAD(NatType, edge->prog_1[self]))) && (((!WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])])) && (WLOAD(NatType, edge->prog_3[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) > WLOAD(NatType, edge->prog_3[self]))) || (WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])]) && (WLOAD(NatType, edge->prog_3[LOAD(node->cand[WLOAD(RefType, edge->v[self])])]) < WLOAD(NatType, edge->prog_3[self])))))) {
		// v.cand := this;
		SetParam<RefType>(WLOAD(RefType, edge->v[self]), node->cand, self, stable);
	}
}

__device__ void execute_Edge_self_loops_to_top(RefType self,
											   bool* stable){
	
	if ((((WLOAD(RefType, edge->v[self]) == WLOAD(RefType, edge->w[self])) && WLOAD(BoolType, node->is_odd[WLOAD(RefType, edge->v[self])])) && ((WLOAD(NatType, node->p[WLOAD(RefType, edge->v[self])]) % 2) == 1))) {
		// v.rho_top := true;
		SetParam<BoolType>(WLOAD(RefType, edge->v[self]), node->rho_top, true, stable);
	}
}

__device__ void execute_Edge_print(RefType self,
								   bool* stable){
		if (self != 0) {
		printf("Edge(%u): v=%u, w=%u, max1=%u, max3=%u, prog_top=%u, prog_1=%u, prog_3=%u\n", self, LOAD(edge->v[self]), LOAD(edge->w[self]), LOAD(edge->max1[self]), LOAD(edge->max3[self]), LOAD(edge->prog_top[self]), LOAD(edge->prog_1[self]), LOAD(edge->prog_3[self]));
	}

}


__global__ void kernel_Node_lift(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = node->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_lift(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Node_count_odd(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = node->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_count_odd(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Node_print_odd(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = node->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_print_odd(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Node_print(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = node->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_print(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Edge_prog(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = edge->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_prog(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Edge_minmax_top(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = edge->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_minmax_top(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Edge_minmax_1(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = edge->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_minmax_1(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Edge_minmax_3(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = edge->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_minmax_3(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Edge_self_loops_to_top(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = edge->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_self_loops_to_top(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
		}
	}

}

__global__ void kernel_Edge_print(int fp_lvl){
	const grid_group grid = this_grid();
	const uint bl_rank = this_thread_block().thread_rank();
	inst_size nrof_instances = edge->nrof_instances();

	__shared__ uint32_t stable[32];
	if(bl_rank < 32){
		stable[bl_rank] = (uint32_t)true;
	}

	__syncthreads();

	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_print(self, (bool*)&stable[bl_rank % 32]);
	}

	if(fp_lvl >= 0){
		__syncthreads();
		if(bl_rank < 32){
			bool stable_reduced = __all_sync(0xffffffff, stable[bl_rank]);
			if(bl_rank == 0 && !stable_reduced){
				clear_stack(fp_lvl);
			}
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
	CHECK(cudaHostRegister(&host_Edge, sizeof(Edge), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_Node, sizeof(Node), cudaHostRegisterDefault));

	inst_size Edge_capacity = structs[0].nrof_instances + 1;
	host_Edge.initialise(&structs[0], Edge_capacity);
	inst_size Node_capacity = structs[1].nrof_instances + 1;
	host_Node.initialise(&structs[1], Node_capacity);

	inst_size max_nrof_executing_instances = max(Node_capacity, Edge_capacity);
	CHECK(cudaDeviceSynchronize());

	Edge * const loc_edge = (Edge*)host_Edge.to_device();
	Node * const loc_node = (Node*)host_Node.to_device();

	CHECK(cudaMemcpyToSymbol(edge, &loc_edge, sizeof(Edge * const)));
	CHECK(cudaMemcpyToSymbol(node, &loc_node, sizeof(Node * const)));

	cudaStream_t kernel_stream;
	CHECK(cudaStreamCreate(&kernel_stream));
	Schedule schedule((void*)launch_kernel, (void*)relaunch_fp_kernel);

	schedule.add_step((void*)kernel_Edge_self_loops_to_top, Edge_capacity, 128);
	schedule.begin_fixpoint();
		schedule.add_step((void*)kernel_Edge_prog, Edge_capacity, 128);
		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_Edge_minmax_top, Edge_capacity, 128);
		schedule.end_fixpoint();

		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_Edge_minmax_1, Edge_capacity, 128);
		schedule.end_fixpoint();

		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_Edge_minmax_3, Edge_capacity, 128);
		schedule.end_fixpoint();

		schedule.add_step((void*)kernel_Node_lift, Node_capacity, 128);
	schedule.end_fixpoint();

	schedule.add_step((void*)kernel_Node_count_odd, Node_capacity, 128);
	schedule.add_step((void*)kernel_Node_print_odd, Node_capacity, 128);	cudaGraphExec_t graph_exec = schedule.instantiate(kernel_stream);
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
