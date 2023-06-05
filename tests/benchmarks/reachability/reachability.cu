#define ATOMIC(T) cuda::atomic<T, cuda::thread_scope_device>
#define STORE(A, B) A.store(B, cuda::memory_order_relaxed)
#define LOAD(A) A.load(cuda::memory_order_relaxed)
#define THREADS_PER_BLOCK 512
#define INSTS_PER_THREAD 32
#define FP_DEPTH 1


#include "ADL.h"
#include "Struct.h"
#include "init_file.h"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
#include <vector>


class Node : public Struct {
public:
	Node (void) : Struct() {}
	
	ATOMIC(BoolType)* reachable;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Node");
		assert (info->parameter_types.size() == 1);
		assert (info->parameter_types[0] == ADL::Bool);
	};

	void** get_parameters(void) {
		return (void**)&reachable;
	}

	size_t child_size(void) {
		return sizeof(Node);
	}

	size_t param_size(uint idx) {
		static size_t sizes[1] = {
			sizeof(ATOMIC(BoolType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(BoolType _reachable,
									   bool step_parity){
		RefType slot = claim_instance2(step_parity);
		STORE(reachable[slot], _reachable);
		return slot;
	}
};

class Edge : public Struct {
public:
	Edge (void) : Struct() {}
	
	ATOMIC(RefType)* n1;
	ATOMIC(RefType)* n2;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Edge");
		assert (info->parameter_types.size() == 2);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&n1;
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

	__device__ RefType create_instance(RefType _n1,
									   RefType _n2,
									   bool step_parity){
		RefType slot = claim_instance2(step_parity);
		STORE(n1[slot], _n1);
		STORE(n2[slot], _n2);
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

__device__ __inline__ bool print_Node(const RefType self){
	bool stable = true;
	if (self != 0) {
		printf("Node(%u): reachable=%u\n", self, LOAD(node->reachable[self]));
	}
	return stable;
}

__device__ __inline__ bool print_Edge(const RefType self){
	bool stable = true;
	if (self != 0) {
		printf("Edge(%u): n1=%u, n2=%u\n", self, LOAD(edge->n1[self]), LOAD(edge->n2[self]));
	}
	return stable;
}

__device__ __inline__ bool Edge_reachability(const RefType self,
											 bool step_parity){
	bool stable = true;
	
	if (LOAD(node->reachable[LOAD(edge->n1[self])])) {
		// n2.reachable := true;
		SetParam(LOAD(edge->n2[self]), node->reachable, true, &stable);
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


		nrof_instances = edge->nrof_instances2(step_parity);
		for(int i = 0; i < INSTS_PER_THREAD; i++){
			self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
			if (self >= nrof_instances) break;

			if (!Edge_reachability(self, step_parity)) {
				stable = false;	
			}				
		}
		step_parity = !step_parity;
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

	host_Edge.initialise(&structs[0], 10000);
	host_Node.initialise(&structs[1], 10000);

	CHECK(cudaDeviceSynchronize());

	Edge * const loc_edge = (Edge*)host_Edge.to_device();
	Node * const loc_node = (Node*)host_Node.to_device();

	CHECK(cudaMemcpyToSymbol(edge, &loc_edge, sizeof(Edge * const)));
	CHECK(cudaMemcpyToSymbol(node, &loc_node, sizeof(Node * const)));

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
