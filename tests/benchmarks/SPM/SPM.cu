#define I_PER_THREAD 8
#define THREADS_PER_BLOCK 256
#define ATOMIC(T) cuda::atomic<T, cuda::thread_scope_device>
#define STORE(A, B) A.store(B, cuda::memory_order_relaxed)
#define LOAD(A) A.load(cuda::memory_order_relaxed)
#define FP_DEPTH 2
#define Node_MASK (1ULL << 0)
#define Edge_MASK (1ULL << 1)
#define Measure_MASK (1ULL << 2)
#define STEP_PARITY(STRUCT) ((bool)(struct_step_parity & STRUCT ## _MASK))
#define TOGGLE_STEP_PARITY(STRUCT) {struct_step_parity ^= STRUCT ## _MASK;}


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
	
	ATOMIC(NatType)* p;
	ATOMIC(BoolType)* owner;
	ATOMIC(RefType)* rho;
	ATOMIC(RefType)* candidate;
	ATOMIC(RefType)* max;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Node");
		assert (info->parameter_types.size() == 5);
		assert (info->parameter_types[0] == ADL::Nat);
		assert (info->parameter_types[1] == ADL::Bool);
		assert (info->parameter_types[2] == ADL::Ref);
		assert (info->parameter_types[3] == ADL::Ref);
		assert (info->parameter_types[4] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&p;
	}

	size_t child_size(void) {
		return sizeof(Node);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[5] = {
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(NatType _p,
									   BoolType _owner,
									   RefType _rho,
									   RefType _candidate,
									   RefType _max,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(p[slot], _p);
		STORE(owner[slot], _owner);
		STORE(rho[slot], _rho);
		STORE(candidate[slot], _candidate);
		STORE(max[slot], _max);
		*stable = false;
		return slot;
	}
};

class Edge : public Struct {
public:
	Edge (void) : Struct() {}
	
	ATOMIC(RefType)* v;
	ATOMIC(RefType)* w;
	ATOMIC(RefType)* m;
	ATOMIC(RefType)* max;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Edge");
		assert (info->parameter_types.size() == 4);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
		assert (info->parameter_types[2] == ADL::Ref);
		assert (info->parameter_types[3] == ADL::Ref);
	};

	void** get_parameters(void) {
		return (void**)&v;
	}

	size_t child_size(void) {
		return sizeof(Edge);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[4] = {
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType)),
			sizeof(ATOMIC(RefType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(RefType _v,
									   RefType _w,
									   RefType _m,
									   RefType _max,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(v[slot], _v);
		STORE(w[slot], _w);
		STORE(m[slot], _m);
		STORE(max[slot], _max);
		*stable = false;
		return slot;
	}
};

class Measure : public Struct {
public:
	Measure (void) : Struct() {}
	
	ATOMIC(BoolType)* top;
	ATOMIC(NatType)* p1;
	ATOMIC(NatType)* p3;

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Measure");
		assert (info->parameter_types.size() == 3);
		assert (info->parameter_types[0] == ADL::Bool);
		assert (info->parameter_types[1] == ADL::Nat);
		assert (info->parameter_types[2] == ADL::Nat);
	};

	void** get_parameters(void) {
		return (void**)&top;
	}

	size_t child_size(void) {
		return sizeof(Measure);
	}

	size_t param_size(uint idx) {
		static const size_t sizes[3] = {
			sizeof(ATOMIC(BoolType)),
			sizeof(ATOMIC(NatType)),
			sizeof(ATOMIC(NatType))
		};
		return sizes[idx];
	}

	__device__ RefType create_instance(BoolType _top,
									   NatType _p1,
									   NatType _p3,
									   bool* stable){
		RefType slot = claim_instance2();
		STORE(top[slot], _top);
		STORE(p1[slot], _p1);
		STORE(p3[slot], _p3);
		*stable = false;
		return slot;
	}
};

using namespace cooperative_groups;

Edge host_Edge = Edge();
Measure host_Measure = Measure();
Node host_Node = Node();

Edge* host_Edge_ptr = &host_Edge;
Measure* host_Measure_ptr = &host_Measure;
Node* host_Node_ptr = &host_Node;

__device__ Edge* __restrict__ edge;
__device__ Measure* __restrict__ measure;
__device__ Node* __restrict__ node;

/* Transform an iter_idx into the fp_stack index
   associated with that operation.
*/
#define FP_SET(X) (X)
#define FP_RESET(X) ((X) + 1 >= 3 ? (X) + 1 - 3 : (X) + 1)
#define FP_READ(X) ((X) + 2 >= 3 ? (X) + 2 - 3 : (X) + 2)

__device__ cuda::atomic<bool, cuda::thread_scope_device> fp_stack[FP_DEPTH][3];

__device__ __inline__ void clear_stack(int lvl, uint8_t* iter_idx) {
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
	for(int i = 0; i < I_PER_THREAD; i++){
		const RefType self = block.size() * (i + grid.block_rank() * I_PER_THREAD) + block.thread_rank();
		if (self >= nrof_instances) break;

		Step(self, stable);
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

__device__ __inline__ void print_Node(const RefType self,
									  bool* stable){
	if (self != 0) {
		printf("Node(%u): p=%u, owner=%u, rho=%u, candidate=%u, max=%u\n", self, LOAD(node->p[self]), LOAD(node->owner[self]), LOAD(node->rho[self]), LOAD(node->candidate[self]), LOAD(node->max[self]));
	}
}

__device__ void Node_reset_candidate(const RefType self,
									 bool* stable){
	
	if (LOAD(node->owner[self])) {
		// candidate.top := true;
		SetParam(LOAD(node->candidate[self]), measure->top, true, stable);
		// candidate.p1 := max.p1;
		SetParam(LOAD(node->candidate[self]), measure->p1, LOAD(measure->p1[LOAD(node->max[self])]), stable);
		// candidate.p3 := max.p3;
		SetParam(LOAD(node->candidate[self]), measure->p3, LOAD(measure->p3[LOAD(node->max[self])]), stable);
	}
	if ((!LOAD(node->owner[self]))) {
		// candidate.top := false;
		SetParam(LOAD(node->candidate[self]), measure->top, false, stable);
		// candidate.p1 := 0;
		SetParam(LOAD(node->candidate[self]), measure->p1, 0, stable);
		// candidate.p3 := 0;
		SetParam(LOAD(node->candidate[self]), measure->p3, 0, stable);
	}
}

__device__ void Node_max_candidate(const RefType self,
								   bool* stable){
	
	BoolType copy = false;
	if (LOAD(measure->top[LOAD(node->candidate[self])])) {
		copy = true;
	}
	if ((!LOAD(measure->top[LOAD(node->candidate[self])]))) {
		if ((LOAD(measure->p1[LOAD(node->candidate[self])]) > LOAD(measure->p1[LOAD(node->rho[self])]))) {
			copy = true;
		}
		if ((LOAD(measure->p1[LOAD(node->candidate[self])]) <= LOAD(measure->p1[LOAD(node->rho[self])]))) {
			if ((LOAD(measure->p3[LOAD(node->candidate[self])]) > LOAD(measure->p3[LOAD(node->rho[self])]))) {
				copy = true;
			}
		}
	}
	if (copy) {
		// rho.top := candidate.top;
		SetParam(LOAD(node->rho[self]), measure->top, LOAD(measure->top[LOAD(node->candidate[self])]), stable);
		// rho.p1 := candidate.p1;
		SetParam(LOAD(node->rho[self]), measure->p1, LOAD(measure->p1[LOAD(node->candidate[self])]), stable);
		// rho.p3 := candidate.p3;
		SetParam(LOAD(node->rho[self]), measure->p3, LOAD(measure->p3[LOAD(node->candidate[self])]), stable);
	}
}

__device__ __inline__ void print_Edge(const RefType self,
									  bool* stable){
	if (self != 0) {
		printf("Edge(%u): v=%u, w=%u, m=%u, max=%u\n", self, LOAD(edge->v[self]), LOAD(edge->w[self]), LOAD(edge->m[self]), LOAD(edge->max[self]));
	}
}

__device__ void Edge_prog(const RefType self,
						  bool* stable){
	
	if (((LOAD(node->p[LOAD(edge->v[self])]) % 2) == 0)) {
		// m.top := w.rho.top;
		SetParam(LOAD(edge->m[self]), measure->top, LOAD(measure->top[LOAD(node->rho[LOAD(edge->w[self])])]), stable);
		if ((!LOAD(measure->top[LOAD(edge->m[self])]))) {
			if ((LOAD(node->p[LOAD(edge->v[self])]) >= 1)) {
				// m.p1 := w.rho.p1;
				SetParam(LOAD(edge->m[self]), measure->p1, LOAD(measure->p1[LOAD(node->rho[LOAD(edge->w[self])])]), stable);
			}
			if ((LOAD(node->p[LOAD(edge->v[self])]) < 1)) {
				// m.p1 := 0;
				SetParam(LOAD(edge->m[self]), measure->p1, 0, stable);
			}
			if ((LOAD(node->p[LOAD(edge->v[self])]) >= 3)) {
				// m.p3 := w.rho.p3;
				SetParam(LOAD(edge->m[self]), measure->p3, LOAD(measure->p3[LOAD(node->rho[LOAD(edge->w[self])])]), stable);
			}
			if ((LOAD(node->p[LOAD(edge->v[self])]) < 3)) {
				// m.p3 := 0;
				SetParam(LOAD(edge->m[self]), measure->p3, 0, stable);
			}
		}
	}
	if (((LOAD(node->p[LOAD(edge->v[self])]) % 2) == 1)) {
		// m.top := w.rho.top;
		SetParam(LOAD(edge->m[self]), measure->top, LOAD(measure->top[LOAD(node->rho[LOAD(edge->w[self])])]), stable);
		if ((!LOAD(measure->top[LOAD(edge->m[self])]))) {
			BoolType incr = false;
			if (((LOAD(node->p[LOAD(edge->v[self])]) >= 3) && (LOAD(measure->p3[LOAD(node->rho[LOAD(edge->w[self])])]) < LOAD(measure->p3[LOAD(edge->max[self])])))) {
				// m.p3 := w.rho.p3 + 1;
				SetParam(LOAD(edge->m[self]), measure->p3, (LOAD(measure->p3[LOAD(node->rho[LOAD(edge->w[self])])]) + 1), stable);
				incr = true;
			}
			if ((!incr)) {
				// m.p3 := 0;
				SetParam(LOAD(edge->m[self]), measure->p3, 0, stable);
				if (((LOAD(node->p[LOAD(edge->v[self])]) >= 1) && (LOAD(measure->p1[LOAD(node->rho[LOAD(edge->w[self])])]) < LOAD(measure->p1[LOAD(edge->max[self])])))) {
					// m.p1 := w.rho.p1 + 1;
					SetParam(LOAD(edge->m[self]), measure->p1, (LOAD(measure->p1[LOAD(node->rho[LOAD(edge->w[self])])]) + 1), stable);
					incr = true;
				}
				if ((!incr)) {
					// m.top := true;
					SetParam(LOAD(edge->m[self]), measure->top, true, stable);
				}
			}
		}
	}
}

__device__ void Edge_top(const RefType self,
						 bool* stable){
	
	if ((LOAD(node->owner[LOAD(edge->v[self])]) && (!LOAD(measure->top[LOAD(edge->m[self])])))) {
		// v.candidate.top := false;
		SetParam(LOAD(node->candidate[LOAD(edge->v[self])]), measure->top, false, stable);
	}
	if (((!LOAD(node->owner[LOAD(edge->v[self])])) && LOAD(measure->top[LOAD(edge->m[self])]))) {
		// v.candidate.top := true;
		SetParam(LOAD(node->candidate[LOAD(edge->v[self])]), measure->top, true, stable);
	}
}

__device__ void Edge_priority_1(const RefType self,
								bool* stable){
	
	if ((!LOAD(measure->top[LOAD(node->candidate[LOAD(edge->v[self])])]))) {
		if (((LOAD(node->owner[LOAD(edge->v[self])]) && (LOAD(measure->p1[LOAD(edge->m[self])]) < LOAD(measure->p1[LOAD(node->candidate[LOAD(edge->v[self])])]))) || ((!LOAD(node->owner[LOAD(edge->v[self])])) && (LOAD(measure->p1[LOAD(edge->m[self])]) > LOAD(measure->p1[LOAD(node->candidate[LOAD(edge->v[self])])]))))) {
			// v.candidate.p1 := m.p1;
			SetParam(LOAD(node->candidate[LOAD(edge->v[self])]), measure->p1, LOAD(measure->p1[LOAD(edge->m[self])]), stable);
		}
	}
}

__device__ void Edge_priority_3(const RefType self,
								bool* stable){
	
	if (((!LOAD(measure->top[LOAD(node->candidate[LOAD(edge->v[self])])])) && (LOAD(measure->p1[LOAD(node->candidate[LOAD(edge->v[self])])]) == LOAD(measure->p1[LOAD(edge->m[self])])))) {
		if (((LOAD(node->owner[LOAD(edge->v[self])]) && (LOAD(measure->p3[LOAD(edge->m[self])]) < LOAD(measure->p3[LOAD(node->candidate[LOAD(edge->v[self])])]))) || ((!LOAD(node->owner[LOAD(edge->v[self])])) && (LOAD(measure->p3[LOAD(edge->m[self])]) > LOAD(measure->p3[LOAD(node->candidate[LOAD(edge->v[self])])]))))) {
			// v.candidate.p3 := m.p3;
			SetParam(LOAD(node->candidate[LOAD(edge->v[self])]), measure->p3, LOAD(measure->p3[LOAD(edge->m[self])]), stable);
		}
	}
}

__device__ __inline__ void print_Measure(const RefType self,
										 bool* stable){
	if (self != 0) {
		printf("Measure(%u): top=%u, p1=%u, p3=%u\n", self, LOAD(measure->top[self]), LOAD(measure->p1[self]), LOAD(measure->p3[self]));
	}
}

__device__ void Measure_init(const RefType self,
							 bool* stable){
	
	BoolType even = true;
	BoolType odd = false;
	RefType max = measure->create_instance(false, 2, 3, stable);
	RefType X = node->create_instance(1, odd, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType X_p = node->create_instance(1, even, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType Y_p = node->create_instance(2, even, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType Y = node->create_instance(2, odd, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType W = node->create_instance(3, even, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType Z = node->create_instance(3, even, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType Z_p = node->create_instance(3, even, measure->create_instance(false, 0, 0, stable), measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e1 = edge->create_instance(X, X, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e2 = edge->create_instance(X, X_p, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e3 = edge->create_instance(X_p, Y, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e4 = edge->create_instance(X_p, Z, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e5 = edge->create_instance(Y, Y_p, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e6 = edge->create_instance(Y, W, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e7 = edge->create_instance(Y_p, Y, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e8 = edge->create_instance(Y_p, X, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e9 = edge->create_instance(W, W, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e10 = edge->create_instance(W, Z, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e11 = edge->create_instance(Z, Z_p, measure->create_instance(false, 0, 0, stable), max, stable);
	RefType e12 = edge->create_instance(Z_p, Z_p, measure->create_instance(false, 0, 0, stable), max, stable);
}


__global__ void schedule_kernel(){
	const grid_group grid = this_grid();
	const thread_block block = this_thread_block();
	const bool is_thread0 = grid.thread_rank() == 0;
	inst_size nrof_instances;
	uint64_t struct_step_parity = 0; // bitmask
	bool stable = true; // Only used to compile steps outside fixpoints
	uint8_t iter_idx[FP_DEPTH] = {0}; // Denotes which fp_stack index ([0, 2]) is currently being set.

	TOGGLE_STEP_PARITY(Measure);
	nrof_instances = measure->nrof_instances2(STEP_PARITY(Measure));
	executeStep<Measure_init>(nrof_instances, grid, block, &stable);
	measure->update_counters(!STEP_PARITY(Measure));

	grid.sync();

	do{
		bool stable = true;
		if (is_thread0){
			/* Resets the next fp_stack index in advance. */
			fp_stack[0][FP_RESET(iter_idx[0])].store(true, cuda::memory_order_relaxed);
		}


		TOGGLE_STEP_PARITY(Edge);
		nrof_instances = edge->nrof_instances2(STEP_PARITY(Edge));
		executeStep<Edge_prog>(nrof_instances, grid, block, &stable);
		edge->update_counters(!STEP_PARITY(Edge));

		grid.sync();

		TOGGLE_STEP_PARITY(Node);
		nrof_instances = node->nrof_instances2(STEP_PARITY(Node));
		executeStep<Node_reset_candidate>(nrof_instances, grid, block, &stable);
		node->update_counters(!STEP_PARITY(Node));

		grid.sync();

		TOGGLE_STEP_PARITY(Edge);
		nrof_instances = edge->nrof_instances2(STEP_PARITY(Edge));
		executeStep<Edge_top>(nrof_instances, grid, block, &stable);
		edge->update_counters(!STEP_PARITY(Edge));

		grid.sync();

		do{
			bool stable = true;
			if (is_thread0){
				/* Resets the next fp_stack index in advance. */
				fp_stack[1][FP_RESET(iter_idx[1])].store(true, cuda::memory_order_relaxed);
			}


			TOGGLE_STEP_PARITY(Edge);
			nrof_instances = edge->nrof_instances2(STEP_PARITY(Edge));
			executeStep<Edge_priority_1>(nrof_instances, grid, block, &stable);
			edge->update_counters(!STEP_PARITY(Edge));
			if(!stable){
				clear_stack(1, iter_idx);
			}
			/* The next index to set is the one that has been reset. */
			iter_idx[1] = FP_RESET(iter_idx[1]);
			grid.sync();
		} while(!fp_stack[1][FP_READ(iter_idx[1])].load(cuda::memory_order_relaxed));


		do{
			bool stable = true;
			if (is_thread0){
				/* Resets the next fp_stack index in advance. */
				fp_stack[1][FP_RESET(iter_idx[1])].store(true, cuda::memory_order_relaxed);
			}


			TOGGLE_STEP_PARITY(Edge);
			nrof_instances = edge->nrof_instances2(STEP_PARITY(Edge));
			executeStep<Edge_priority_3>(nrof_instances, grid, block, &stable);
			edge->update_counters(!STEP_PARITY(Edge));
			if(!stable){
				clear_stack(1, iter_idx);
			}
			/* The next index to set is the one that has been reset. */
			iter_idx[1] = FP_RESET(iter_idx[1]);
			grid.sync();
		} while(!fp_stack[1][FP_READ(iter_idx[1])].load(cuda::memory_order_relaxed));


		TOGGLE_STEP_PARITY(Node);
		nrof_instances = node->nrof_instances2(STEP_PARITY(Node));
		executeStep<Node_max_candidate>(nrof_instances, grid, block, &stable);
		node->update_counters(!STEP_PARITY(Node));
		if(!stable){
			clear_stack(0, iter_idx);
		}
		/* The next index to set is the one that has been reset. */
		iter_idx[0] = FP_RESET(iter_idx[0]);
		grid.sync();
	} while(!fp_stack[0][FP_READ(iter_idx[0])].load(cuda::memory_order_relaxed));

}


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}

	std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 4194304);
	CHECK(cudaHostRegister(&host_Edge, sizeof(Edge), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_Measure, sizeof(Measure), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_Node, sizeof(Node), cudaHostRegisterDefault));

	host_Edge.initialise(&structs[0], 100);
	host_Measure.initialise(&structs[1], 100);
	host_Node.initialise(&structs[2], 100);

	CHECK(cudaDeviceSynchronize());

	Edge * const loc_edge = (Edge*)host_Edge.to_device();
	Measure * const loc_measure = (Measure*)host_Measure.to_device();
	Node * const loc_node = (Node*)host_Node.to_device();

	CHECK(cudaMemcpyToSymbol(edge, &loc_edge, sizeof(Edge * const)));
	CHECK(cudaMemcpyToSymbol(measure, &loc_measure, sizeof(Measure * const)));
	CHECK(cudaMemcpyToSymbol(node, &loc_node, sizeof(Node * const)));

	cuda::atomic<bool, cuda::thread_scope_device>* fp_stack_address;
	cudaGetSymbolAddress((void **)&fp_stack_address, fp_stack);
	CHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 3 * sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));

	void* schedule_kernel_args[] = {};
	auto dims = ADL::get_launch_dims(38, (void*)schedule_kernel);

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
