#define FP_DEPTH 2
#define THREADS_PER_BLOCK 512


#include "ADL.h"
#include "Struct.h"
#include "fp_manager.h"
#include "init_file.h"
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <stdio.h>
#include <vector>


using namespace cooperative_groups;
class Edge : public Struct {
public:
	Edge (void) : Struct() {}
	
	union {
		void* parameters[2];
		struct {
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* s;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* t;
		};
	};

	void assert_correct_info(InitFile::StructInfo* info) {
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

	size_t param_size(uint idx) {
		static size_t sizes[2] = {
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>)
		};
		return sizes[idx];
	}

	__host__ __device__ RefType create_instance(ADL::RefType _s, ADL::RefType _t) {
		RefType slot = claim_instance();
		s[slot].store(_s, cuda::memory_order_relaxed);
		t[slot].store(_t, cuda::memory_order_relaxed);
		return slot;
	}
};

class Node : public Struct {
public:
	Node (void) : Struct() {}
	
	union {
		void* parameters[5];
		struct {
			cuda::atomic<ADL::BoolType, cuda::thread_scope_device>* in_scc;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* fwd;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* bwd;
			cuda::atomic<ADL::BoolType, cuda::thread_scope_device>* fwd_valid;
			cuda::atomic<ADL::BoolType, cuda::thread_scope_device>* bwd_valid;
		};
	};

	void assert_correct_info(InitFile::StructInfo* info) {
		assert (info->name == "Node");
		assert (info->parameter_types.size() == 5);
		assert (info->parameter_types[0] == ADL::Bool);
		assert (info->parameter_types[1] == ADL::Ref);
		assert (info->parameter_types[2] == ADL::Ref);
		assert (info->parameter_types[3] == ADL::Bool);
		assert (info->parameter_types[4] == ADL::Bool);
	};

	void** get_parameters(void) {
		return parameters;
	}

	size_t child_size(void) {
		return sizeof(Node);
	}

	size_t param_size(uint idx) {
		static size_t sizes[5] = {
			sizeof(cuda::atomic<ADL::BoolType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::BoolType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::BoolType, cuda::thread_scope_device>)
		};
		return sizes[idx];
	}

	__host__ __device__ RefType create_instance(ADL::BoolType _in_scc, ADL::RefType _fwd, ADL::RefType _bwd, ADL::BoolType _fwd_valid, ADL::BoolType _bwd_valid) {
		RefType slot = claim_instance();
		in_scc[slot].store(_in_scc, cuda::memory_order_relaxed);
		fwd[slot].store(_fwd, cuda::memory_order_relaxed);
		bwd[slot].store(_bwd, cuda::memory_order_relaxed);
		fwd_valid[slot].store(_fwd_valid, cuda::memory_order_relaxed);
		bwd_valid[slot].store(_bwd_valid, cuda::memory_order_relaxed);
		return slot;
	}
};



FPManager host_FP = FPManager(FP_DEPTH);
FPManager* device_FP;

Edge host_Edge = Edge();
Node host_Node = Node();

Edge* host_Edge_ptr = &host_Edge;
Node* host_Node_ptr = &host_Node;

Edge* gm_Edge;
Node* gm_Node;



__global__ void Edge_propagate(FPManager* FP,
							   inst_size nrof_instances,
							   Edge* const edge,
							   Node* const node){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
	
		if ((!(node->in_scc[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) && (!(node->in_scc[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)))) {
			if ((node->fwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) < (node->fwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) {
				/* t.fwd = (node->fwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) */
				par_owner = (edge->t[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::RefType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = (node->fwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
					if (prev_val != new_val) {
						node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if ((node->bwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) < (node->bwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) {
				/* s.bwd = (node->bwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) */
				par_owner = (edge->s[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::RefType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = (node->bwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
					if (prev_val != new_val) {
						node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
		}
	}
	
}

__global__ void Edge_determine_valid_frontier(FPManager* FP,
											  inst_size nrof_instances,
											  Edge* const edge,
											  Node* const node){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
	
		if ((!(node->in_scc[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) && (!(node->in_scc[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)))) {
			if ((node->fwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) != (node->fwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) {
				/* s.fwd.fwd_valid = false */
				par_owner = (node->fwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::BoolType prev_val = node->fwd_valid[par_owner].load(cuda::memory_order_relaxed);
					ADL::BoolType new_val = false;
					if (prev_val != new_val) {
						node->fwd_valid[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if ((node->bwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) != (node->bwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) {
				/* t.bwd.bwd_valid = false */
				par_owner = (node->bwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::BoolType prev_val = node->bwd_valid[par_owner].load(cuda::memory_order_relaxed);
					ADL::BoolType new_val = false;
					if (prev_val != new_val) {
						node->bwd_valid[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
		}
	}
	
}

__global__ void Node_init(FPManager* FP,
						  inst_size nrof_instances,
						  Edge* const edge,
						  Node* const node){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
	
		/* in_scc = false */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->in_scc[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = false;
			if (prev_val != new_val) {
				node->in_scc[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		/* fwd = self */
		par_owner = self;
		if(par_owner != 0){
			ADL::RefType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
			ADL::RefType new_val = self;
			if (prev_val != new_val) {
				node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		/* bwd = self */
		par_owner = self;
		if(par_owner != 0){
			ADL::RefType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
			ADL::RefType new_val = self;
			if (prev_val != new_val) {
				node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		/* fwd_valid = true */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->fwd_valid[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = true;
			if (prev_val != new_val) {
				node->fwd_valid[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		/* bwd_valid = true */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->bwd_valid[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = true;
			if (prev_val != new_val) {
				node->bwd_valid[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

	}
	
}

__global__ void Node_remove_valid_sccs(FPManager* FP,
									   inst_size nrof_instances,
									   Edge* const edge,
									   Node* const node){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
	
		/* in_scc = ((((node->fwd[self].load(cuda::memory_order_relaxed)) == (node->bwd[self].load(cuda::memory_order_relaxed))) && (node->fwd_valid[node->fwd[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) && (node->bwd_valid[node->bwd[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->in_scc[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = ((((node->fwd[self].load(cuda::memory_order_relaxed)) == (node->bwd[self].load(cuda::memory_order_relaxed))) && (node->fwd_valid[node->fwd[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) && (node->bwd_valid[node->bwd[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)));
			if (prev_val != new_val) {
				node->in_scc[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

	}
	
}

__global__ void Node_reset_fwd_bwd(FPManager* FP,
								   inst_size nrof_instances,
								   Edge* const edge,
								   Node* const node){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
	
		/* fwd_valid = true */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->fwd_valid[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = true;
			if (prev_val != new_val) {
				node->fwd_valid[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		/* bwd_valid = true */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->bwd_valid[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = true;
			if (prev_val != new_val) {
				node->bwd_valid[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		if (!(node->in_scc[self].load(cuda::memory_order_relaxed))) {
			if (node->in_scc[node->fwd[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) {
				/* fwd = self */
				par_owner = self;
				if(par_owner != 0){
					ADL::RefType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if (node->in_scc[node->bwd[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) {
				/* bwd = self */
				par_owner = self;
				if(par_owner != 0){
					ADL::RefType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
		}
	}
	
}
__global__ void Edge_print(Edge* edge,
						   inst_size nrof_instances){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
		if (self != 0) {
			printf("Edge(%u): s=%u, t=%u\n", self, edge->s[self].load(cuda::memory_order_relaxed), edge->t[self].load(cuda::memory_order_relaxed));
		}
	}
}

__global__ void Node_print(Node* node,
						   inst_size nrof_instances){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += grid.num_threads()){
		if (self != 0) {
			printf("Node(%u): in_scc=%u, fwd=%u, bwd=%u, fwd_valid=%u, bwd_valid=%u\n", self, node->in_scc[self].load(cuda::memory_order_relaxed), node->fwd[self].load(cuda::memory_order_relaxed), node->bwd[self].load(cuda::memory_order_relaxed), node->fwd_valid[self].load(cuda::memory_order_relaxed), node->bwd_valid[self].load(cuda::memory_order_relaxed));
		}
	}
}


void launch_Edge_propagate() {
	inst_size nrof_instances = host_Edge.nrof_instances();
	void* Edge_propagate_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_Edge,
		&gm_Node
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Edge_propagate);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Edge_propagate,
			std::get<0>(dims),
			std::get<1>(dims),
			Edge_propagate_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Edge_determine_valid_frontier() {
	inst_size nrof_instances = host_Edge.nrof_instances();
	void* Edge_determine_valid_frontier_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_Edge,
		&gm_Node
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Edge_determine_valid_frontier);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Edge_determine_valid_frontier,
			std::get<0>(dims),
			std::get<1>(dims),
			Edge_determine_valid_frontier_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Node_init() {
	inst_size nrof_instances = host_Node.nrof_instances();
	void* Node_init_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_Edge,
		&gm_Node
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Node_init);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Node_init,
			std::get<0>(dims),
			std::get<1>(dims),
			Node_init_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Node_remove_valid_sccs() {
	inst_size nrof_instances = host_Node.nrof_instances();
	void* Node_remove_valid_sccs_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_Edge,
		&gm_Node
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Node_remove_valid_sccs);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Node_remove_valid_sccs,
			std::get<0>(dims),
			std::get<1>(dims),
			Node_remove_valid_sccs_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Node_reset_fwd_bwd() {
	inst_size nrof_instances = host_Node.nrof_instances();
	void* Node_reset_fwd_bwd_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_Edge,
		&gm_Node
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Node_reset_fwd_bwd);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Node_reset_fwd_bwd,
			std::get<0>(dims),
			std::get<1>(dims),
			Node_reset_fwd_bwd_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}


int main(int argc, char **argv) {
	if (argc != 2) {
		printf("Supply a .init file.\n");
		exit(1);
	}
	
	std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
	CHECK(cudaHostRegister(&host_Edge, sizeof(Edge), cudaHostRegisterDefault));
	CHECK(cudaHostRegister(&host_Node, sizeof(Node), cudaHostRegisterDefault));

	host_Edge.initialise(&structs[0], 100);
	host_Node.initialise(&structs[1], 100);

	CHECK(cudaDeviceSynchronize());

	gm_Edge = (Edge*)host_Edge.to_device();
	gm_Node = (Node*)host_Node.to_device();



	size_t printf_size;
	cudaDeviceGetLimit(&printf_size, cudaLimitPrintfFifoSize);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2 * printf_size);


	host_FP.push();
	device_FP = host_FP.to_device();




	launch_Node_init();



	host_FP.copy_from(device_FP);
	host_FP.push();
	do{
		host_FP.reset();
		host_FP.copy_to(device_FP);
		host_FP.copy_from(device_FP);
		host_FP.push();
		do{
			host_FP.reset();
			host_FP.copy_to(device_FP);

			launch_Edge_propagate();



			host_FP.copy_from(device_FP);
			if(!host_FP.done()) host_FP.clear();
		}
		while(!host_FP.done());
		host_FP.pop();
		host_FP.copy_to(device_FP);


		launch_Edge_determine_valid_frontier();




		launch_Node_remove_valid_sccs();




		launch_Node_reset_fwd_bwd();



		host_FP.copy_from(device_FP);
		if(!host_FP.done()) host_FP.clear();
	}
	while(!host_FP.done());
	host_FP.pop();
	host_FP.copy_to(device_FP);

	Node_print<<<(host_Node.nrof_instances() + 512 - 1)/512, 512>>>(gm_Node, host_Node.nrof_instances());
	CHECK(cudaDeviceSynchronize());



	
}
