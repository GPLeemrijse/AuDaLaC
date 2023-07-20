#define ATOMIC(T) cuda::atomic<T, cuda::thread_scope_device>
#define STORE(A, V) A.store(V, cuda::memory_order_release)
#define LOAD(A) A.load(cuda::memory_order_acquire)

#define WLOAD(T, A) *((T*)&A)
#define ACQLOAD(A) A.load(cuda::memory_order_acquire)
#define WSTORE(T, A, V) *((T*)&A) = V
#define RELSTORE(A, V) A.store(V, cuda::memory_order_release)

#define Node_MASK (((uint16_t)1) << 0)
#define Edge_MASK (((uint16_t)1) << 1)
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
#define FP_DEPTH 2
__device__ cuda::atomic<bool, cuda::thread_scope_device> fp_stack[FP_DEPTH];

__device__ void clear_stack(int lvl) {
	while(lvl >= 0){
		fp_stack[lvl--].store(false, cuda::memory_order_relaxed);
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

__device__ void execute_Node_lift(RefType self,
								  bool* stable){
	
	// dirty := false;
	SetParam<BoolType>(self, node->dirty, false, stable);
	if ((LOAD(node->cand[self]) != 0)) {
		if (((((LOAD(edge->prog_top[LOAD(node->cand[self])]) != LOAD(node->rho_top[self])) && LOAD(edge->prog_top[LOAD(node->cand[self])])) || ((LOAD(edge->prog_top[LOAD(node->cand[self])]) == LOAD(node->rho_top[self])) && (LOAD(edge->prog_1[LOAD(node->cand[self])]) > LOAD(node->rho_1[self])))) || (((LOAD(edge->prog_top[LOAD(node->cand[self])]) == LOAD(node->rho_top[self])) && (LOAD(edge->prog_1[LOAD(node->cand[self])]) == LOAD(node->rho_1[self]))) && (LOAD(edge->prog_3[LOAD(node->cand[self])]) > LOAD(node->rho_3[self]))))) {
			// rho_top := cand.prog_top;
			SetParam<BoolType>(self, node->rho_top, LOAD(edge->prog_top[LOAD(node->cand[self])]), stable);
			// rho_1 := cand.prog_1;
			SetParam<NatType>(self, node->rho_1, LOAD(edge->prog_1[LOAD(node->cand[self])]), stable);
			// rho_3 := cand.prog_3;
			SetParam<NatType>(self, node->rho_3, LOAD(edge->prog_3[LOAD(node->cand[self])]), stable);
			// dirty := true;
			SetParam<BoolType>(self, node->dirty, true, stable);
		}
	}
}

__device__ void execute_Node_count_odd(RefType self,
									   bool* stable){
	
	if (((self != 0) && LOAD(node->rho_top[self]))) {
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
	
	if (LOAD(node->dirty[LOAD(edge->w[self])])) {
		BoolType m_top = LOAD(node->rho_top[LOAD(edge->w[self])]);
		NatType m_1 = 0;
		if ((LOAD(node->p[LOAD(edge->v[self])]) >= 1)) {
			m_1 = LOAD(node->rho_1[LOAD(edge->w[self])]);
		}
		NatType m_3 = 0;
		if ((LOAD(node->p[LOAD(edge->v[self])]) >= 3)) {
			m_3 = LOAD(node->rho_3[LOAD(edge->w[self])]);
		}
		if (((LOAD(node->p[LOAD(edge->v[self])]) % 2) == 1)) {
			BoolType increased = m_top;
			if (((LOAD(node->p[LOAD(edge->v[self])]) >= 3) && (!increased))) {
				NatType new_3 = 0;
				if ((m_3 < LOAD(edge->max3[self]))) {
					new_3 = (m_3 + 1);
					increased = true;
				}
				m_3 = new_3;
			}
			if (((LOAD(node->p[LOAD(edge->v[self])]) >= 1) && (!increased))) {
				NatType new_1 = 0;
				if ((m_1 < LOAD(edge->max1[self]))) {
					new_1 = (m_1 + 1);
					increased = true;
				}
				m_1 = new_1;
			}
			if ((!increased)) {
				m_top = true;
			}
		}
		if ((LOAD(node->cand[LOAD(edge->v[self])]) == self)) {
			BoolType changed = (((m_top != LOAD(edge->prog_top[self])) || (LOAD(edge->prog_1[self]) != m_1)) || (LOAD(edge->prog_3[self]) != m_3));
			if (changed) {
				// v.cand := null;
				SetParam<RefType>(LOAD(edge->v[self]), node->cand, 0, stable);
			}
		}
		// prog_top := m_top;
		SetParam<BoolType>(self, edge->prog_top, m_top, stable);
		// prog_1 := m_1;
		SetParam<NatType>(self, edge->prog_1, m_1, stable);
		// prog_3 := m_3;
		SetParam<NatType>(self, edge->prog_3, m_3, stable);
	}
}

__device__ void execute_Edge_minmax_top(RefType self,
										bool* stable){
	
	if (((LOAD(node->cand[LOAD(edge->v[self])]) == 0) || ((((!LOAD(node->is_odd[LOAD(edge->v[self])])) && LOAD(edge->prog_top[LOAD(node->cand[LOAD(edge->v[self])])])) && (!LOAD(edge->prog_top[self]))) || ((LOAD(node->is_odd[LOAD(edge->v[self])]) && (!LOAD(edge->prog_top[LOAD(node->cand[LOAD(edge->v[self])])]))) && LOAD(edge->prog_top[self]))))) {
		// v.cand := this;
		SetParam<RefType>(LOAD(edge->v[self]), node->cand, self, stable);
	}
}

__device__ void execute_Edge_minmax_1(RefType self,
									  bool* stable){
	
	if (((LOAD(edge->prog_top[LOAD(node->cand[LOAD(edge->v[self])])]) == LOAD(edge->prog_top[self])) && (((!LOAD(node->is_odd[LOAD(edge->v[self])])) && (LOAD(edge->prog_1[LOAD(node->cand[LOAD(edge->v[self])])]) > LOAD(edge->prog_1[self]))) || (LOAD(node->is_odd[LOAD(edge->v[self])]) && (LOAD(edge->prog_1[LOAD(node->cand[LOAD(edge->v[self])])]) < LOAD(edge->prog_1[self])))))) {
		// v.cand := this;
		SetParam<RefType>(LOAD(edge->v[self]), node->cand, self, stable);
	}
}

__device__ void execute_Edge_minmax_3(RefType self,
									  bool* stable){
	
	if ((((LOAD(edge->prog_top[LOAD(node->cand[LOAD(edge->v[self])])]) == LOAD(edge->prog_top[self])) && (LOAD(edge->prog_1[LOAD(node->cand[LOAD(edge->v[self])])]) == LOAD(edge->prog_1[self]))) && (((!LOAD(node->is_odd[LOAD(edge->v[self])])) && (LOAD(edge->prog_3[LOAD(node->cand[LOAD(edge->v[self])])]) > LOAD(edge->prog_3[self]))) || (LOAD(node->is_odd[LOAD(edge->v[self])]) && (LOAD(edge->prog_3[LOAD(node->cand[LOAD(edge->v[self])])]) < LOAD(edge->prog_3[self])))))) {
		// v.cand := this;
		SetParam<RefType>(LOAD(edge->v[self]), node->cand, self, stable);
	}
}

__device__ void execute_Edge_self_loops_to_top(RefType self,
											   bool* stable){
	
	if ((((LOAD(edge->v[self]) == LOAD(edge->w[self])) && LOAD(node->is_odd[LOAD(edge->v[self])])) && ((LOAD(node->p[LOAD(edge->v[self])]) % 2) == 1))) {
		// v.rho_top := true;
		SetParam<BoolType>(LOAD(edge->v[self]), node->rho_top, true, stable);
	}
}

__device__ void execute_Edge_print(RefType self,
								   bool* stable){
		if (self != 0) {
		printf("Edge(%u): v=%u, w=%u, max1=%u, max3=%u, prog_top=%u, prog_1=%u, prog_3=%u\n", self, LOAD(edge->v[self]), LOAD(edge->w[self]), LOAD(edge->max1[self]), LOAD(edge->max3[self]), LOAD(edge->prog_top[self]), LOAD(edge->prog_1[self]), LOAD(edge->prog_3[self]));
	}

}


__global__ void schedule_kernel(){
	const grid_group grid = this_grid();
	const thread_block block = this_thread_block();
	uint16_t struct_step_parity = 0; // bitmask
	bool stable = true; // Only used to compile steps outside fixpoints


	TOGGLE_STEP_PARITY(Edge);
	executeStep<execute_Edge_self_loops_to_top>(edge->nrof_instances2(STEP_PARITY(Edge)), grid, block, &stable);
	edge->update_counters(!STEP_PARITY(Edge));

	grid.sync();

	do{
		bool stable = true;
		grid.sync();
		if (grid.thread_rank() == 0)
			fp_stack[0].store(true, cuda::memory_order_relaxed);
		grid.sync();


		TOGGLE_STEP_PARITY(Edge);
		executeStep<execute_Edge_prog>(edge->nrof_instances2(STEP_PARITY(Edge)), grid, block, &stable);
		edge->update_counters(!STEP_PARITY(Edge));

		grid.sync();

		do{
			bool stable = true;
			grid.sync();
			if (grid.thread_rank() == 0)
				fp_stack[1].store(true, cuda::memory_order_relaxed);
			grid.sync();


			TOGGLE_STEP_PARITY(Edge);
			executeStep<execute_Edge_minmax_top>(edge->nrof_instances2(STEP_PARITY(Edge)), grid, block, &stable);
			edge->update_counters(!STEP_PARITY(Edge));
			if(!stable)
				clear_stack(1);
			grid.sync();
		} while(!fp_stack[1].load(cuda::memory_order_relaxed));


		do{
			bool stable = true;
			grid.sync();
			if (grid.thread_rank() == 0)
				fp_stack[1].store(true, cuda::memory_order_relaxed);
			grid.sync();


			TOGGLE_STEP_PARITY(Edge);
			executeStep<execute_Edge_minmax_1>(edge->nrof_instances2(STEP_PARITY(Edge)), grid, block, &stable);
			edge->update_counters(!STEP_PARITY(Edge));
			if(!stable)
				clear_stack(1);
			grid.sync();
		} while(!fp_stack[1].load(cuda::memory_order_relaxed));


		do{
			bool stable = true;
			grid.sync();
			if (grid.thread_rank() == 0)
				fp_stack[1].store(true, cuda::memory_order_relaxed);
			grid.sync();


			TOGGLE_STEP_PARITY(Edge);
			executeStep<execute_Edge_minmax_3>(edge->nrof_instances2(STEP_PARITY(Edge)), grid, block, &stable);
			edge->update_counters(!STEP_PARITY(Edge));
			if(!stable)
				clear_stack(1);
			grid.sync();
		} while(!fp_stack[1].load(cuda::memory_order_relaxed));


		TOGGLE_STEP_PARITY(Node);
		executeStep<execute_Node_lift>(node->nrof_instances2(STEP_PARITY(Node)), grid, block, &stable);
		node->update_counters(!STEP_PARITY(Node));
		if(!stable)
			clear_stack(0);
		grid.sync();
	} while(!fp_stack[0].load(cuda::memory_order_relaxed));


	TOGGLE_STEP_PARITY(Node);
	executeStep<execute_Node_count_odd>(node->nrof_instances2(STEP_PARITY(Node)), grid, block, &stable);
	node->update_counters(!STEP_PARITY(Node));

	grid.sync();

	TOGGLE_STEP_PARITY(Node);
	executeStep<execute_Node_print_odd>(node->nrof_instances2(STEP_PARITY(Node)), grid, block, &stable);
	node->update_counters(!STEP_PARITY(Node));
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


	void* schedule_kernel_args[] = {};
	auto dims = get_launch_dims(max_nrof_executing_instances, (void*)schedule_kernel, true);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);


	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)schedule_kernel,
			std::get<0>(dims),
			std::get<1>(dims),
			schedule_kernel_args
		)
	);


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0;
	cudaEventElapsedTime(&ms, start, stop);
	printf("Total walltime GPU: %0.6f ms\n", ms);

}
