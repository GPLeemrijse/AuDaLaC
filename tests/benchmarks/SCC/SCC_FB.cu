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


class NodeSet : public Struct {
public:
	NodeSet (void) : Struct() {}
	
	ATOMIC(RefType)* pivot_f_b;
	ATOMIC(RefType)* pivot_f_nb;
	ATOMIC(RefType)* pivot_nf_b;
	ATOMIC(RefType)* pivot_nf_nb;
	ATOMIC(BoolType)* scc;
	ATOMIC(RefType)* f_and_b;
	ATOMIC(RefType)* not_f_and_b;
	ATOMIC(RefType)* f_and_not_b;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "NodeSet");
		assert (info->parameter_types.size() == 8);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
		assert (info->parameter_types[2] == ADL::Ref);
		assert (info->parameter_types[3] == ADL::Ref);
		assert (info->parameter_types[4] == ADL::Bool);
		assert (info->parameter_types[5] == ADL::Ref);
		assert (info->parameter_types[6] == ADL::Ref);
		assert (info->parameter_types[7] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&pivot_f_b;
	}

	size_t child_size(void) {
		return sizeof(NodeSet);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[8] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _pivot_f_b,
									   RefType _pivot_f_nb,
									   RefType _pivot_nf_b,
									   RefType _pivot_nf_nb,
									   BoolType _scc,
									   RefType _f_and_b,
									   RefType _not_f_and_b,
									   RefType _f_and_not_b,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(pivot_f_b[slot], _pivot_f_b);
		STORE(pivot_f_nb[slot], _pivot_f_nb);
		STORE(pivot_nf_b[slot], _pivot_nf_b);
		STORE(pivot_nf_nb[slot], _pivot_nf_nb);
		STORE(scc[slot], _scc);
		STORE(f_and_b[slot], _f_and_b);
		STORE(not_f_and_b[slot], _not_f_and_b);
		STORE(f_and_not_b[slot], _f_and_not_b);
		*stable = false;
		return slot;
	}
};

class Node : public Struct {
public:
	Node (void) : Struct() {}
	
	ATOMIC(RefType)* set;
	ATOMIC(BoolType)* fwd;
	ATOMIC(BoolType)* bwd;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Node");
		assert (info->parameter_types.size() == 3);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Bool);
		assert (info->parameter_types[2] == ADL::Bool);
	};

	void** get_parameters(void) {
		return (void**)&set;
	}

	size_t child_size(void) {
		return sizeof(Node);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[3] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(BoolType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _set,
									   BoolType _fwd,
									   BoolType _bwd,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(set[slot], _set);
		STORE(fwd[slot], _fwd);
		STORE(bwd[slot], _bwd);
		*stable = false;
		return slot;
	}
};

class Edge : public Struct {
public:
	Edge (void) : Struct() {}
	
	ATOMIC(RefType)* s;
	ATOMIC(RefType)* t;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Edge");
		assert (info->parameter_types.size() == 2);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&s;
	}

	size_t child_size(void) {
		return sizeof(Edge);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[2] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _s,
									   RefType _t,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(s[slot], _s);
		STORE(t[slot], _t);
		*stable = false;
		return slot;
	}
};

using namespace cooperative_groups;


Edge host_Edge = Edge();
Node host_Node = Node();
NodeSet host_NodeSet = NodeSet();

Edge* host_Edge_ptr = &host_Edge;
Node* host_Node_ptr = &host_Node;
NodeSet* host_NodeSet_ptr = &host_NodeSet;

__device__ Edge* __restrict__ edge;
__device__ Node* __restrict__ node;
__device__ NodeSet* __restrict__ nodeset;

__device__ uint nrof_components = 1;
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

__device__ void execute_NodeSet_allocate_sets(RefType self,
											  bool* stable){
	
	if ((WLOAD(RefType, nodeset->pivot_f_b[self]) != 0)) {
		if ((WLOAD(RefType, nodeset->pivot_nf_nb[self]) == 0)) {
			// f_and_b := this;
			WeakSetParam<RefType>(self, nodeset->f_and_b, self, stable);
			// scc := true;
			WeakSetParam<BoolType>(self, nodeset->scc, true, stable);
			// pivot_f_b := null;
			WeakSetParam<RefType>(self, nodeset->pivot_f_b, 0, stable);
		}
		if ((WLOAD(RefType, nodeset->pivot_nf_nb[self]) != 0)) {
			// f_and_b := NodeSet(null, null, null, null, true, null, null, null);
			WeakSetParam<RefType>(self, nodeset->f_and_b, nodeset->create_instance(0, 0, 0, 0, true, 0, 0, 0, stable), stable);
			// pivot_f_b := null;
			WeakSetParam<RefType>(self, nodeset->pivot_f_b, 0, stable);
			atomicInc(&nrof_components, 0xffffffff);
		}
	}
	if ((WLOAD(RefType, nodeset->pivot_f_nb[self]) != 0)) {
		// f_and_not_b := NodeSet(null, pivot_f_nb, null, null, false, null, null, null);
		WeakSetParam<RefType>(self, nodeset->f_and_not_b, nodeset->create_instance(0, WLOAD(RefType, nodeset->pivot_f_nb[self]), 0, 0, false, 0, 0, 0, stable), stable);
		// pivot_f_nb := null;
		WeakSetParam<RefType>(self, nodeset->pivot_f_nb, 0, stable);
		atomicInc(&nrof_components, 0xffffffff);
	}
	if ((WLOAD(RefType, nodeset->pivot_nf_b[self]) != 0)) {
		// not_f_and_b := NodeSet(null, null, pivot_nf_b, null, false, null, null, null);
		WeakSetParam<RefType>(self, nodeset->not_f_and_b, nodeset->create_instance(0, 0, WLOAD(RefType, nodeset->pivot_nf_b[self]), 0, false, 0, 0, 0, stable), stable);
		// pivot_nf_b := null;
		WeakSetParam<RefType>(self, nodeset->pivot_nf_b, 0, stable);
		atomicInc(&nrof_components, 0xffffffff);
	}
}

__device__ void execute_NodeSet_initialise_pivot_fwd_bwd(RefType self,
														 bool* stable){
	
	if ((!WLOAD(BoolType, nodeset->scc[self]))) {
		// pivot_f_b.fwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_f_b[self]), node->fwd, true, stable);
		// pivot_f_b.bwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_f_b[self]), node->bwd, true, stable);
		// pivot_f_b := null;
		WeakSetParam<RefType>(self, nodeset->pivot_f_b, 0, stable);
		// pivot_f_nb.fwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_f_nb[self]), node->fwd, true, stable);
		// pivot_f_nb.bwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_f_nb[self]), node->bwd, true, stable);
		// pivot_f_nb := null;
		WeakSetParam<RefType>(self, nodeset->pivot_f_nb, 0, stable);
		// pivot_nf_b.fwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_nf_b[self]), node->fwd, true, stable);
		// pivot_nf_b.bwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_nf_b[self]), node->bwd, true, stable);
		// pivot_nf_b := null;
		WeakSetParam<RefType>(self, nodeset->pivot_nf_b, 0, stable);
		// pivot_nf_nb.fwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_nf_nb[self]), node->fwd, true, stable);
		// pivot_nf_nb.bwd := true;
		SetParam<BoolType>(WLOAD(RefType, nodeset->pivot_nf_nb[self]), node->bwd, true, stable);
		// pivot_nf_nb := null;
		WeakSetParam<RefType>(self, nodeset->pivot_nf_nb, 0, stable);
	}
}

__device__ void execute_NodeSet_print_nrof_sets(RefType self,
												bool* stable){
	
	if ((self == 0)) {
		printf("Nrof components: %u\n", nrof_components);
	}
}

__device__ void execute_NodeSet_print(RefType self,
									  bool* stable){
		if (self != 0) {
		printf("NodeSet(%u): pivot_f_b=%u, pivot_f_nb=%u, pivot_nf_b=%u, pivot_nf_nb=%u, scc=%u, f_and_b=%u, not_f_and_b=%u, f_and_not_b=%u\n", self, LOAD(nodeset->pivot_f_b[self]), LOAD(nodeset->pivot_f_nb[self]), LOAD(nodeset->pivot_nf_b[self]), LOAD(nodeset->pivot_nf_nb[self]), LOAD(nodeset->scc[self]), LOAD(nodeset->f_and_b[self]), LOAD(nodeset->not_f_and_b[self]), LOAD(nodeset->f_and_not_b[self]));
	}

}

__device__ void execute_Node_pivots_nominate(RefType self,
											 bool* stable){
	
	if ((!WLOAD(BoolType, nodeset->scc[WLOAD(RefType, node->set[self])]))) {
		BoolType f = WLOAD(BoolType, node->fwd[self]);
		BoolType b = WLOAD(BoolType, node->bwd[self]);
		if ((f && b)) {
			// set.pivot_f_b := this;
			SetParam<RefType>(WLOAD(RefType, node->set[self]), nodeset->pivot_f_b, self, stable);
		}
		if ((f && (!b))) {
			// set.pivot_f_nb := this;
			SetParam<RefType>(WLOAD(RefType, node->set[self]), nodeset->pivot_f_nb, self, stable);
		}
		if (((!f) && b)) {
			// set.pivot_nf_b := this;
			SetParam<RefType>(WLOAD(RefType, node->set[self]), nodeset->pivot_nf_b, self, stable);
		}
		if (((!f) && (!b))) {
			// set.pivot_nf_nb := this;
			SetParam<RefType>(WLOAD(RefType, node->set[self]), nodeset->pivot_nf_nb, self, stable);
		}
	}
}

__device__ void execute_Node_divide_into_sets_reset_fwd_bwd(RefType self,
															bool* stable){
	
	BoolType f = WLOAD(BoolType, node->fwd[self]);
	BoolType b = WLOAD(BoolType, node->bwd[self]);
	if ((f && b)) {
		// set := set.f_and_b;
		WeakSetParam<RefType>(self, node->set, WLOAD(RefType, nodeset->f_and_b[WLOAD(RefType, node->set[self])]), stable);
	}
	if (((!f) && b)) {
		// set := set.not_f_and_b;
		WeakSetParam<RefType>(self, node->set, WLOAD(RefType, nodeset->not_f_and_b[WLOAD(RefType, node->set[self])]), stable);
	}
	if ((f && (!b))) {
		// set := set.f_and_not_b;
		WeakSetParam<RefType>(self, node->set, WLOAD(RefType, nodeset->f_and_not_b[WLOAD(RefType, node->set[self])]), stable);
	}
	// fwd := false;
	WeakSetParam<BoolType>(self, node->fwd, false, stable);
	// bwd := false;
	WeakSetParam<BoolType>(self, node->bwd, false, stable);
}

__device__ void execute_Node_print(RefType self,
								   bool* stable){
		if (self != 0) {
		printf("Node(%u): set=%u, fwd=%u, bwd=%u\n", self, LOAD(node->set[self]), LOAD(node->fwd[self]), LOAD(node->bwd[self]));
	}

}

__device__ void execute_Edge_compute_fwd_bwd(RefType self,
											 bool* stable){
	
	if ((WLOAD(RefType, node->set[WLOAD(RefType, edge->t[self])]) == WLOAD(RefType, node->set[WLOAD(RefType, edge->s[self])]))) {
		if (LOAD(node->fwd[WLOAD(RefType, edge->s[self])])) {
			// t.fwd := true;
			SetParam<BoolType>(WLOAD(RefType, edge->t[self]), node->fwd, true, stable);
		}
		if (LOAD(node->bwd[WLOAD(RefType, edge->t[self])])) {
			// s.bwd := true;
			SetParam<BoolType>(WLOAD(RefType, edge->s[self]), node->bwd, true, stable);
		}
	}
}

__device__ void execute_Edge_print(RefType self,
								   bool* stable){
		if (self != 0) {
		printf("Edge(%u): s=%u, t=%u\n", self, LOAD(edge->s[self]), LOAD(edge->t[self]));
	}

}


__global__ void kernel_NodeSet_allocate_sets(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = nodeset->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_NodeSet_allocate_sets(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_NodeSet_initialise_pivot_fwd_bwd(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = nodeset->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_NodeSet_initialise_pivot_fwd_bwd(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_NodeSet_print_nrof_sets(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = nodeset->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_NodeSet_print_nrof_sets(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_NodeSet_print(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = nodeset->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_NodeSet_print(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_Node_pivots_nominate(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = node->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_pivots_nominate(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_Node_divide_into_sets_reset_fwd_bwd(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = node->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_divide_into_sets_reset_fwd_bwd(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_Node_print(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = node->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Node_print(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_Edge_compute_fwd_bwd(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = edge->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_compute_fwd_bwd(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
}

__global__ void kernel_Edge_print(int fp_lvl){
	const grid_group grid = this_grid();
	inst_size nrof_instances = edge->nrof_instances();
		bool stable = true;
	bool* stable_ptr = &stable;
	RefType self = grid.thread_rank();
	if (self < nrof_instances){
		execute_Edge_print(self, stable_ptr);
	}
	
	if(!stable && fp_lvl >= 0)
		clear_stack(fp_lvl);
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

__global__ void update_nrof_NodeSet(){
	nodeset->set_active_to_created();
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
	CHECK(cudaHostRegister(&host_NodeSet, sizeof(NodeSet), cudaHostRegisterDefault));

	inst_size Edge_capacity = structs[0].nrof_instances + 1;
	host_Edge.initialise(&structs[0], Edge_capacity);
	inst_size Node_capacity = structs[1].nrof_instances + 1;
	host_Node.initialise(&structs[1], Node_capacity);
	inst_size NodeSet_capacity = 5000000;
	host_NodeSet.initialise(&structs[2], NodeSet_capacity);

	inst_size max_nrof_executing_instances = max(NodeSet_capacity, max(Node_capacity, Edge_capacity));
	CHECK(cudaDeviceSynchronize());

	Edge * const loc_edge = (Edge*)host_Edge.to_device();
	Node * const loc_node = (Node*)host_Node.to_device();
	NodeSet * const loc_nodeset = (NodeSet*)host_NodeSet.to_device();

	CHECK(cudaMemcpyToSymbol(edge, &loc_edge, sizeof(Edge * const)));
	CHECK(cudaMemcpyToSymbol(node, &loc_node, sizeof(Node * const)));
	CHECK(cudaMemcpyToSymbol(nodeset, &loc_nodeset, sizeof(NodeSet * const)));

	cudaStream_t kernel_stream;
	CHECK(cudaStreamCreate(&kernel_stream));
	Schedule schedule((void*)launch_kernel, (void*)relaunch_fp_kernel);

	schedule.begin_fixpoint();
		schedule.add_step((void*)kernel_Node_pivots_nominate, Node_capacity, 128);
		schedule.add_step((void*)kernel_NodeSet_allocate_sets, NodeSet_capacity, 128);
		schedule.add_step((void*)update_nrof_NodeSet, 1, 0);
		schedule.add_step((void*)kernel_Node_divide_into_sets_reset_fwd_bwd, Node_capacity, 128);
		schedule.add_step((void*)kernel_NodeSet_initialise_pivot_fwd_bwd, NodeSet_capacity, 128);
		schedule.begin_fixpoint();
			schedule.add_step((void*)kernel_Edge_compute_fwd_bwd, Edge_capacity, 128);
		schedule.end_fixpoint();

	schedule.end_fixpoint();

	schedule.add_step((void*)kernel_NodeSet_print_nrof_sets, NodeSet_capacity, 128);	cudaGraphExec_t graph_exec = schedule.instantiate(kernel_stream);
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
