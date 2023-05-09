#define ATOMIC(T) cuda::atomic<T, cuda::thread_scope_device>
#define STORE(A, B) A.store(B, cuda::memory_order_relaxed)
#define LOAD(A) A.load(cuda::memory_order_relaxed)
#define THREADS_PER_BLOCK 512
#define INSTS_PER_THREAD 32
#define FP_DEPTH 2


#include "ADL.h"
#include "Struct.h"
#include "init_file.h"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
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
		static size_t sizes[8] = {
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
									   bool step_parity){
		RefType slot = claim_instance2(step_parity);
		STORE(pivot_f_b[slot], _pivot_f_b);
		STORE(pivot_f_nb[slot], _pivot_f_nb);
		STORE(pivot_nf_b[slot], _pivot_nf_b);
		STORE(pivot_nf_nb[slot], _pivot_nf_nb);
		STORE(scc[slot], _scc);
		STORE(f_and_b[slot], _f_and_b);
		STORE(not_f_and_b[slot], _not_f_and_b);
		STORE(f_and_not_b[slot], _f_and_not_b);
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
		static size_t sizes[3] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(BoolType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _set,
									   BoolType _fwd,
									   BoolType _bwd,
									   bool step_parity){
		RefType slot = claim_instance2(step_parity);
		STORE(set[slot], _set);
		STORE(fwd[slot], _fwd);
		STORE(bwd[slot], _bwd);
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
		static size_t sizes[2] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _s,
									   RefType _t,
									   bool step_parity){
		RefType slot = claim_instance2(step_parity);
		STORE(s[slot], _s);
		STORE(t[slot], _t);
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

__device__ bool fp_stack[FP_DEPTH][2];

__device__ __inline__ void clear_stack(int lvl, bool iteration_parity) {
	/*	For the first lvl, only clear the iteration_parity bool.
		The first !iteration_parity bool is being set to true in advance.
	*/
	fp_stack[lvl--][iteration_parity] = false;

	while(lvl >= 0){
		fp_stack[lvl][iteration_parity] = false;
		fp_stack[lvl][!iteration_parity] = false;
		lvl--;
	}
}

__device__ __inline__ void initialize_stack() {
	for(int i = 0; i < FP_DEPTH; i++){
		fp_stack[i][0] = true;
		fp_stack[i][1] = true;
	}
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

__device__ __inline__ bool print_NodeSet(const RefType self){
	bool stable = true;
	if (self != 0) {
		printf("NodeSet(%u): pivot_f_b=%u, pivot_f_nb=%u, pivot_nf_b=%u, pivot_nf_nb=%u, scc=%u, f_and_b=%u, not_f_and_b=%u, f_and_not_b=%u\n", self, LOAD(nodeset->pivot_f_b[self]), LOAD(nodeset->pivot_f_nb[self]), LOAD(nodeset->pivot_nf_b[self]), LOAD(nodeset->pivot_nf_nb[self]), LOAD(nodeset->scc[self]), LOAD(nodeset->f_and_b[self]), LOAD(nodeset->not_f_and_b[self]), LOAD(nodeset->f_and_not_b[self]));
	}
	return stable;
}

__device__ __inline__ bool NodeSet_allocate_sets(const RefType self,
												 bool step_parity){
	bool stable = true;
	
	if ((LOAD(nodeset->pivot_f_b[self]) != (RefType)0)) {
		if ((LOAD(nodeset->pivot_nf_nb[self]) == (RefType)0)) {
			// f_and_b := this;
			SetParam(self, nodeset->f_and_b, self, &stable);
			// scc := true;
			SetParam(self, nodeset->scc, true, &stable);
			// pivot_f_b := null;
			SetParam(self, nodeset->pivot_f_b, (RefType)0, &stable);
		}
		if ((LOAD(nodeset->pivot_nf_nb[self]) != (RefType)0)) {
			// f_and_b := NodeSet(null, null, null, null, true, null, null, null);
			SetParam(self, nodeset->f_and_b, nodeset->create_instance((RefType)0, (RefType)0, (RefType)0, (RefType)0, true, (RefType)0, (RefType)0, (RefType)0, step_parity), &stable);
			// pivot_f_b := null;
			SetParam(self, nodeset->pivot_f_b, (RefType)0, &stable);
		}
	}
	if ((LOAD(nodeset->pivot_f_nb[self]) != (RefType)0)) {
		// f_and_not_b := NodeSet(null, pivot_f_nb, null, null, false, null, null, null);
		SetParam(self, nodeset->f_and_not_b, nodeset->create_instance((RefType)0, LOAD(nodeset->pivot_f_nb[self]), (RefType)0, (RefType)0, false, (RefType)0, (RefType)0, (RefType)0, step_parity), &stable);
		// pivot_f_nb := null;
		SetParam(self, nodeset->pivot_f_nb, (RefType)0, &stable);
	}
	if ((LOAD(nodeset->pivot_nf_b[self]) != (RefType)0)) {
		// not_f_and_b := NodeSet(null, null, pivot_nf_b, null, false, null, null, null);
		SetParam(self, nodeset->not_f_and_b, nodeset->create_instance((RefType)0, (RefType)0, LOAD(nodeset->pivot_nf_b[self]), (RefType)0, false, (RefType)0, (RefType)0, (RefType)0, step_parity), &stable);
		// pivot_nf_b := null;
		SetParam(self, nodeset->pivot_nf_b, (RefType)0, &stable);
	}
	return stable;
}

__device__ __inline__ bool NodeSet_initialise_pivot_fwd_bwd(const RefType self,
															bool step_parity){
	bool stable = true;
	
	if ((!LOAD(nodeset->scc[self]))) {
		// pivot_f_b.fwd := true;
		SetParam(LOAD(nodeset->pivot_f_b[self]), node->fwd, true, &stable);
		// pivot_f_b.bwd := true;
		SetParam(LOAD(nodeset->pivot_f_b[self]), node->bwd, true, &stable);
		// pivot_f_b := null;
		SetParam(self, nodeset->pivot_f_b, (RefType)0, &stable);
		// pivot_f_nb.fwd := true;
		SetParam(LOAD(nodeset->pivot_f_nb[self]), node->fwd, true, &stable);
		// pivot_f_nb.bwd := true;
		SetParam(LOAD(nodeset->pivot_f_nb[self]), node->bwd, true, &stable);
		// pivot_f_nb := null;
		SetParam(self, nodeset->pivot_f_nb, (RefType)0, &stable);
		// pivot_nf_b.fwd := true;
		SetParam(LOAD(nodeset->pivot_nf_b[self]), node->fwd, true, &stable);
		// pivot_nf_b.bwd := true;
		SetParam(LOAD(nodeset->pivot_nf_b[self]), node->bwd, true, &stable);
		// pivot_nf_b := null;
		SetParam(self, nodeset->pivot_nf_b, (RefType)0, &stable);
		// pivot_nf_nb.fwd := true;
		SetParam(LOAD(nodeset->pivot_nf_nb[self]), node->fwd, true, &stable);
		// pivot_nf_nb.bwd := true;
		SetParam(LOAD(nodeset->pivot_nf_nb[self]), node->bwd, true, &stable);
		// pivot_nf_nb := null;
		SetParam(self, nodeset->pivot_nf_nb, (RefType)0, &stable);
	}
	return stable;
}

__device__ __inline__ bool print_Node(const RefType self){
	bool stable = true;
	if (self != 0) {
		printf("Node(%u): set=%u, fwd=%u, bwd=%u\n", self, LOAD(node->set[self]), LOAD(node->fwd[self]), LOAD(node->bwd[self]));
	}
	return stable;
}

__device__ __inline__ bool Node_pivots_nominate(const RefType self,
												bool step_parity){
	bool stable = true;
	
	if ((!LOAD(nodeset->scc[LOAD(node->set[self])]))) {
		BoolType f = LOAD(node->fwd[self]);
		BoolType b = LOAD(node->bwd[self]);
		if ((f && b)) {
			// set.pivot_f_b := this;
			SetParam(LOAD(node->set[self]), nodeset->pivot_f_b, self, &stable);
		}
		if ((f && (!b))) {
			// set.pivot_f_nb := this;
			SetParam(LOAD(node->set[self]), nodeset->pivot_f_nb, self, &stable);
		}
		if (((!f) && b)) {
			// set.pivot_nf_b := this;
			SetParam(LOAD(node->set[self]), nodeset->pivot_nf_b, self, &stable);
		}
		if (((!f) && (!b))) {
			// set.pivot_nf_nb := this;
			SetParam(LOAD(node->set[self]), nodeset->pivot_nf_nb, self, &stable);
		}
	}
	return stable;
}

__device__ __inline__ bool Node_divide_into_sets_reset_fwd_bwd(const RefType self,
															   bool step_parity){
	bool stable = true;
	
	BoolType f = LOAD(node->fwd[self]);
	BoolType b = LOAD(node->bwd[self]);
	if ((f && b)) {
		// set := set.f_and_b;
		SetParam(self, node->set, LOAD(nodeset->f_and_b[LOAD(node->set[self])]), &stable);
	}
	if (((!f) && b)) {
		// set := set.not_f_and_b;
		SetParam(self, node->set, LOAD(nodeset->not_f_and_b[LOAD(node->set[self])]), &stable);
	}
	if ((f && (!b))) {
		// set := set.f_and_not_b;
		SetParam(self, node->set, LOAD(nodeset->f_and_not_b[LOAD(node->set[self])]), &stable);
	}
	// fwd := false;
	SetParam(self, node->fwd, false, &stable);
	// bwd := false;
	SetParam(self, node->bwd, false, &stable);
	return stable;
}

__device__ __inline__ bool print_Edge(const RefType self){
	bool stable = true;
	if (self != 0) {
		printf("Edge(%u): s=%u, t=%u\n", self, LOAD(edge->s[self]), LOAD(edge->t[self]));
	}
	return stable;
}

__device__ __inline__ bool Edge_compute_fwd_bwd(const RefType self,
												bool step_parity){
	bool stable = true;
	
	if ((LOAD(node->set[LOAD(edge->t[self])]) == LOAD(node->set[LOAD(edge->s[self])]))) {
		if (LOAD(node->fwd[LOAD(edge->s[self])])) {
			// t.fwd := true;
			SetParam(LOAD(edge->t[self]), node->fwd, true, &stable);
		}
		if (LOAD(node->bwd[LOAD(edge->t[self])])) {
			// s.bwd := true;
			SetParam(LOAD(edge->s[self]), node->bwd, true, &stable);
		}
	}
	return stable;
}


__global__ void schedule_kernel(){
	const grid_group grid = this_grid();
	const thread_block block = this_thread_block();
	const uint in_grid_rank = grid.thread_rank();
	const uint in_block_rank = block.thread_rank();
	const uint block_idx = grid.block_rank();
	const uint block_size = block.size();
	inst_size nrof_instances;
	bool step_parity = false;
	RefType self;
	bool iteration_parity[FP_DEPTH] = {false};

	do{
		iteration_parity[0] = !iteration_parity[0];
		bool stable = true;
		if (in_grid_rank == 0)
			fp_stack[0][!iteration_parity[0]] = true;


		nrof_instances = node->nrof_instances2(step_parity);
		for(int i = 0; i < INSTS_PER_THREAD; i++){
			self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
			if (self >= nrof_instances) break;

			if (!Node_pivots_nominate(self, step_parity)) {
				stable = false;	
			}				
		}
		step_parity = !step_parity;

		grid.sync();

		nrof_instances = nodeset->nrof_instances2(step_parity);
		for(int i = 0; i < INSTS_PER_THREAD; i++){
			self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
			if (self >= nrof_instances) break;

			if (!NodeSet_allocate_sets(self, step_parity)) {
				stable = false;	
			}				
		}
		step_parity = !step_parity;

		grid.sync();

		nrof_instances = node->nrof_instances2(step_parity);
		for(int i = 0; i < INSTS_PER_THREAD; i++){
			self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
			if (self >= nrof_instances) break;

			if (!Node_divide_into_sets_reset_fwd_bwd(self, step_parity)) {
				stable = false;	
			}				
		}
		step_parity = !step_parity;

		grid.sync();

		nrof_instances = nodeset->nrof_instances2(step_parity);
		for(int i = 0; i < INSTS_PER_THREAD; i++){
			self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
			if (self >= nrof_instances) break;

			if (!NodeSet_initialise_pivot_fwd_bwd(self, step_parity)) {
				stable = false;	
			}				
		}
		step_parity = !step_parity;

		grid.sync();

		do{
			iteration_parity[1] = !iteration_parity[1];
			bool stable = true;
			if (in_grid_rank == 0)
				fp_stack[1][!iteration_parity[1]] = true;


			nrof_instances = edge->nrof_instances2(step_parity);
			for(int i = 0; i < INSTS_PER_THREAD; i++){
				self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
				if (self >= nrof_instances) break;

				if (!Edge_compute_fwd_bwd(self, step_parity)) {
					stable = false;	
				}				
			}
			step_parity = !step_parity;
			if(!stable)
				clear_stack(1, iteration_parity[1]);
			grid.sync();
		} while(!fp_stack[1][iteration_parity[1]]);

		if(!stable)
			clear_stack(0, iteration_parity[0]);
		grid.sync();
	} while(!fp_stack[0][iteration_parity[0]]);


	nrof_instances = node->nrof_instances2(step_parity);
	for(int i = 0; i < INSTS_PER_THREAD; i++){
		self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
		if (self >= nrof_instances) break;

		print_Node(self);				
	}
	step_parity = !step_parity;
}


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}

	std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2097152);
	CHECK(cudaHostRegister(&host_Edge, sizeof(Edge), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_Node, sizeof(Node), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_NodeSet, sizeof(NodeSet), cudaHostRegisterDefault));

	host_Edge.initialise(&structs[0], 10000);
	host_Node.initialise(&structs[1], 10000);
	host_NodeSet.initialise(&structs[2], 10000);

	CHECK(cudaDeviceSynchronize());

	Edge * const loc_edge = (Edge*)host_Edge.to_device();
	Node * const loc_node = (Node*)host_Node.to_device();
	NodeSet * const loc_nodeset = (NodeSet*)host_NodeSet.to_device();

	CHECK(cudaMemcpyToSymbol(edge, &loc_edge, sizeof(Edge * const)));
	CHECK(cudaMemcpyToSymbol(node, &loc_node, sizeof(Node * const)));
	CHECK(cudaMemcpyToSymbol(nodeset, &loc_nodeset, sizeof(NodeSet * const)));

	bool* fp_stack_address;
	cudaGetSymbolAddress((void **)&fp_stack_address, fp_stack);
	CHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 2 * sizeof(bool)));

	void* schedule_kernel_args[] = {};
	auto dims = ADL::get_launch_dims(10000, (void*)schedule_kernel);

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
