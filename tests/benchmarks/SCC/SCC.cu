#include "ADL.h"
#include "Struct.h"
#include "fp_manager.h"
#include "init_file.h"
#include <stdio.h>
#include <vector>


#define FP_DEPTH 2
#define SET_PARAM(P, V, T, I) ({if (I != 0) { T read_val = P; T write_val = V; if (read_val != write_val) {P = write_val; FP->set();}}})


class NodeSet : public Struct {
public:
	NodeSet (void) : Struct() {}
	
	union {
		void* parameters[5];
		struct {
			ADL::RefType* pivot;
			ADL::BoolType* scc;
			ADL::RefType* f_and_b;
			ADL::RefType* not_f_and_b;
			ADL::RefType* f_and_not_b;
		};
	};

	void assertCorrectInfo(InitFile::StructInfo* info) {
		assert (info->name == "NodeSet");
		assert (info->parameter_types.size() == 5);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Bool);
		assert (info->parameter_types[2] == ADL::Ref);
		assert (info->parameter_types[3] == ADL::Ref);
		assert (info->parameter_types[4] == ADL::Ref);
	};

	void** get_parameters(void) {
		return parameters;
	}

	size_t child_size(void) {
		return sizeof(NodeSet);
	}

	__host__ __device__ RefType create_instance(ADL::RefType _pivot, ADL::BoolType _scc, ADL::RefType _f_and_b, ADL::RefType _not_f_and_b, ADL::RefType _f_and_not_b) {
		RefType slot = claim_instance();
		pivot[slot] = _pivot;
		scc[slot] = _scc;
		f_and_b[slot] = _f_and_b;
		not_f_and_b[slot] = _not_f_and_b;
		f_and_not_b[slot] = _f_and_not_b;
		return slot;
	}
};

class Node : public Struct {
public:
	Node (void) : Struct() {}
	
	union {
		void* parameters[3];
		struct {
			ADL::RefType* set;
			ADL::BoolType* fwd;
			ADL::BoolType* bwd;
		};
	};

	void assertCorrectInfo(InitFile::StructInfo* info) {
		assert (info->name == "Node");
		assert (info->parameter_types.size() == 3);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Bool);
		assert (info->parameter_types[2] == ADL::Bool);
	};

	void** get_parameters(void) {
		return parameters;
	}

	size_t child_size(void) {
		return sizeof(Node);
	}

	__host__ __device__ RefType create_instance(ADL::RefType _set, ADL::BoolType _fwd, ADL::BoolType _bwd) {
		RefType slot = claim_instance();
		set[slot] = _set;
		fwd[slot] = _fwd;
		bwd[slot] = _bwd;
		return slot;
	}
};

class Edge : public Struct {
public:
	Edge (void) : Struct() {}
	
	union {
		void* parameters[2];
		struct {
			ADL::RefType* s;
			ADL::RefType* t;
		};
	};

	void assertCorrectInfo(InitFile::StructInfo* info) {
		assert (info->name == "Edge");
		assert (info->parameter_types.size() == 2);
		assert (info->parameter_types[0] == ADL::Ref);
		assert (info->parameter_types[1] == ADL::Ref);
	};

	void** get_parameters(void) {
		return parameters;
	}

	size_t child_size(void) {
		return sizeof(Edge);
	}

	__host__ __device__ RefType create_instance(ADL::RefType _s, ADL::RefType _t) {
		RefType slot = claim_instance();
		s[slot] = _s;
		t[slot] = _t;
		return slot;
	}
};








__global__ void NodeSet_divide_into_sets_reset_pivot(FPManager* FP, NodeSet* const nodeset, Node* const node, Edge* const edge){
	grid_group grid = this_grid();
	RefType self = blockDim.x * blockIdx.x + threadIdx.x;
	if(!nodeset->is_active(self)) { return; }

	SET_PARAM(node->pivot[self], 0, RefType, self);
}

__global__ void NodeSet_pivot_win_allocate_sets(FPManager* FP, NodeSet* const nodeset, Node* const node, Edge* const edge, NodeSet* const host_nodeset){
	grid_group grid = this_grid();
	RefType self = blockDim.x * blockIdx.x + threadIdx.x;
	if(!nodeset->is_active(self)) { return; }

	if (node->pivot[self] == 0) {
		SET_PARAM(bool->scc[self], true, BoolType, self);
	}
	if (node->pivot[self] != 0) {
		SET_PARAM(bool->fwd[node->pivot[self]], true, BoolType, node->pivot[self]);
		SET_PARAM(bool->bwd[node->pivot[self]], true, BoolType, node->pivot[self]);
	}
	if (!bool->scc[self]) {
		RefType intermediate = nodeset->create_instance(0, false, 0, 0, 0);
		bool->scc[intermediate] = true;
		SET_PARAM(nodeset->f_and_b[self], intermediate, RefType, self);
		SET_PARAM(nodeset->f_and_not_b[self], nodeset->create_instance(0, false, 0, 0, 0), RefType, self);
		SET_PARAM(nodeset->not_f_and_b[self], nodeset->create_instance(0, false, 0, 0, 0), RefType, self);
	}

	grid.sync();

	if (self == 0) {
		nodeset->sync_nrof_instances(host_nodeset);
	}

}

__global__ void Node_pivot_nominate(FPManager* FP, NodeSet* const nodeset, Node* const node, Edge* const edge){
	grid_group grid = this_grid();
	RefType self = blockDim.x * blockIdx.x + threadIdx.x;
	if(!node->is_active(self)) { return; }

	if (!bool->scc[nodeset->set[self]]) {
		SET_PARAM(node->pivot[nodeset->set[self]], self, RefType, nodeset->set[self]);
	}
}

__global__ void Node_divide_into_sets_reset_pivot(FPManager* FP, NodeSet* const nodeset, Node* const node, Edge* const edge){
	grid_group grid = this_grid();
	RefType self = blockDim.x * blockIdx.x + threadIdx.x;
	if(!node->is_active(self)) { return; }

	if (bool->fwd[self] && bool->bwd[self]) {
		SET_PARAM(nodeset->set[self], nodeset->f_and_b[nodeset->set[self]], RefType, self);
	}
	if ((!bool->fwd[self]) && bool->bwd[self]) {
		SET_PARAM(nodeset->set[self], nodeset->not_f_and_b[nodeset->set[self]], RefType, self);
	}
	if (bool->fwd[self] && (!bool->bwd[self])) {
		SET_PARAM(nodeset->set[self], nodeset->f_and_not_b[nodeset->set[self]], RefType, self);
	}
	SET_PARAM(bool->fwd[self], false, BoolType, self);
	SET_PARAM(bool->bwd[self], false, BoolType, self);
}

__global__ void Edge_compute_fwd_bwd(FPManager* FP, NodeSet* const nodeset, Node* const node, Edge* const edge){
	grid_group grid = this_grid();
	RefType self = blockDim.x * blockIdx.x + threadIdx.x;
	if(!edge->is_active(self)) { return; }

	if (nodeset->set[node->t[self]] == nodeset->set[node->s[self]]) {
		if bool->fwd[node->s[self]] {
			SET_PARAM(bool->fwd[node->t[self]], true, BoolType, node->t[self]);
		}
		if bool->bwd[node->t[self]] {
			SET_PARAM(bool->bwd[node->s[self]], true, BoolType, node->s[self]);
		}
	}
}


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}
	
	std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
	Edge host_Edge = Edge();
	Node host_Node = Node();
	NodeSet host_NodeSet = NodeSet();

	host_Edge.initialise(&structs[0], 100);
	host_Node.initialise(&structs[1], 100);
	host_NodeSet.initialise(&structs[2], 100);

	CHECK(cudaDeviceSynchronize());

	Edge* gm_Edge = (Edge*)host_Edge.to_device();
	Node* gm_Node = (Node*)host_Node.to_device();
	NodeSet* gm_NodeSet = (NodeSet*)host_NodeSet.to_device();



	FPManager host_FP = FPManager(FP_DEPTH); // initially not done
	FPManager* device_FP = host_FP.to_device();

	host_FP.push();
	do{
		host_FP.reset();
		host_FP.copy_to(device_FP);
		void* Node_pivot_nominate_args[] = {
	,
			
};
		CHECK(
			cudaLaunchCooperativeKernel(
				(void*)Node_pivot_nominate,
				(host_Node.nrof_instances() + 512 - 1)/512,
				512,
				Node_pivot_nominate_args
			)
		);
		CHECK(cudaDeviceSynchronize());

		void* NodeSet_pivot_win_allocate_sets_args[] = {
	,
			
};
		CHECK(
			cudaLaunchCooperativeKernel(
				(void*)NodeSet_pivot_win_allocate_sets,
				(host_NodeSet.nrof_instances() + 512 - 1)/512,
				512,
				NodeSet_pivot_win_allocate_sets_args
			)
		);
		CHECK(cudaDeviceSynchronize());

		host_FP.push();
		do{
			host_FP.reset();
			host_FP.copy_to(device_FP);
			void* Edge_compute_fwd_bwd_args[] = {
	,
				
};
			CHECK(
				cudaLaunchCooperativeKernel(
					(void*)Edge_compute_fwd_bwd,
					(host_Edge.nrof_instances() + 512 - 1)/512,
					512,
					Edge_compute_fwd_bwd_args
				)
			);
			CHECK(cudaDeviceSynchronize());

			host_FP.copy_from(device_FP);
			if(!host_FP.done()) host_FP.clear();
		}
		while(!host_FP.done());
		host_FP.pop();

		void* NodeSet_divide_into_sets_reset_pivot_args[] = {
	,
			
};
		CHECK(
			cudaLaunchCooperativeKernel(
				(void*)NodeSet_divide_into_sets_reset_pivot,
				(host_NodeSet.nrof_instances() + 512 - 1)/512,
				512,
				NodeSet_divide_into_sets_reset_pivot_args
			)
		);
		CHECK(cudaDeviceSynchronize());
		void* Node_divide_into_sets_reset_pivot_args[] = {
	,
			
};
		CHECK(
			cudaLaunchCooperativeKernel(
				(void*)Node_divide_into_sets_reset_pivot,
				(host_Node.nrof_instances() + 512 - 1)/512,
				512,
				Node_divide_into_sets_reset_pivot_args
			)
		);
		CHECK(cudaDeviceSynchronize());

		host_FP.copy_from(device_FP);
		if(!host_FP.done()) host_FP.clear();
	}
	while(!host_FP.done());
	host_FP.pop();



	
}
