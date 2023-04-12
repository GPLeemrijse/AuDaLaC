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
class NodeSet : public Struct {
public:
	NodeSet (void) : Struct() {}
	
	union {
		void* parameters[8];
		struct {
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* pivot_f_b;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* pivot_f_nb;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* pivot_nf_b;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* pivot_nf_nb;
			cuda::atomic<ADL::BoolType, cuda::thread_scope_device>* scc;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* f_and_b;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* not_f_and_b;
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* f_and_not_b;
		};
	};

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
		return parameters;
	}

	size_t child_size(void) {
		return sizeof(NodeSet);
	}

	size_t param_size(uint idx) {
		static size_t sizes[8] = {
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::BoolType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>)
		};
		return sizes[idx];
	}

	__host__ __device__ RefType create_instance(ADL::RefType _pivot_f_b, ADL::RefType _pivot_f_nb, ADL::RefType _pivot_nf_b, ADL::RefType _pivot_nf_nb, ADL::BoolType _scc, ADL::RefType _f_and_b, ADL::RefType _not_f_and_b, ADL::RefType _f_and_not_b) {
		RefType slot = claim_instance();
		pivot_f_b[slot].store(_pivot_f_b, cuda::memory_order_relaxed);
		pivot_f_nb[slot].store(_pivot_f_nb, cuda::memory_order_relaxed);
		pivot_nf_b[slot].store(_pivot_nf_b, cuda::memory_order_relaxed);
		pivot_nf_nb[slot].store(_pivot_nf_nb, cuda::memory_order_relaxed);
		scc[slot].store(_scc, cuda::memory_order_relaxed);
		f_and_b[slot].store(_f_and_b, cuda::memory_order_relaxed);
		not_f_and_b[slot].store(_not_f_and_b, cuda::memory_order_relaxed);
		f_and_not_b[slot].store(_f_and_not_b, cuda::memory_order_relaxed);
		return slot;
	}
};

class Node : public Struct {
public:
	Node (void) : Struct() {}
	
	union {
		void* parameters[3];
		struct {
			cuda::atomic<ADL::RefType, cuda::thread_scope_device>* set;
			cuda::atomic<ADL::BoolType, cuda::thread_scope_device>* fwd;
			cuda::atomic<ADL::BoolType, cuda::thread_scope_device>* bwd;
		};
	};

	void assert_correct_info(InitFile::StructInfo* info) {
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

	size_t param_size(uint idx) {
		static size_t sizes[3] = {
			sizeof(cuda::atomic<ADL::RefType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::BoolType, cuda::thread_scope_device>),
			sizeof(cuda::atomic<ADL::BoolType, cuda::thread_scope_device>)
		};
		return sizes[idx];
	}

	__host__ __device__ RefType create_instance(ADL::RefType _set, ADL::BoolType _fwd, ADL::BoolType _bwd) {
		RefType slot = claim_instance();
		set[slot].store(_set, cuda::memory_order_relaxed);
		fwd[slot].store(_fwd, cuda::memory_order_relaxed);
		bwd[slot].store(_bwd, cuda::memory_order_relaxed);
		return slot;
	}
};

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



FPManager host_FP = FPManager(FP_DEPTH);
FPManager* device_FP;

Edge host_Edge = Edge();
Node host_Node = Node();
NodeSet host_NodeSet = NodeSet();

Edge* host_Edge_ptr = &host_Edge;
Node* host_Node_ptr = &host_Node;
NodeSet* host_NodeSet_ptr = &host_NodeSet;

Edge* gm_Edge;
Node* gm_Node;
NodeSet* gm_NodeSet;



__global__ void NodeSet_allocate_sets(FPManager* FP,
									  inst_size nrof_instances,
									  NodeSet* const nodeset,
									  Node* const node,
									  Edge* const edge,
									  NodeSet* const host_nodeset){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
	
		if ((nodeset->pivot_f_b[self].load(cuda::memory_order_relaxed)) != 0) {
			if ((nodeset->pivot_nf_nb[self].load(cuda::memory_order_relaxed)) == 0) {
				/* f_and_b = self */
				par_owner = self;
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->f_and_b[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						nodeset->f_and_b[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

				/* scc = true */
				par_owner = self;
				if(par_owner != 0){
					ADL::BoolType prev_val = nodeset->scc[par_owner].load(cuda::memory_order_relaxed);
					ADL::BoolType new_val = true;
					if (prev_val != new_val) {
						nodeset->scc[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

				/* pivot_f_b = 0 */
				par_owner = self;
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->pivot_f_b[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = 0;
					if (prev_val != new_val) {
						nodeset->pivot_f_b[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if ((nodeset->pivot_nf_nb[self].load(cuda::memory_order_relaxed)) != 0) {
				/* f_and_b = nodeset->create_instance(0, 0, 0, 0, true, 0, 0, 0) */
				par_owner = self;
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->f_and_b[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = nodeset->create_instance(0, 0, 0, 0, true, 0, 0, 0);
					if (prev_val != new_val) {
						nodeset->f_and_b[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

				/* pivot_f_b = 0 */
				par_owner = self;
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->pivot_f_b[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = 0;
					if (prev_val != new_val) {
						nodeset->pivot_f_b[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
		}
		if ((nodeset->pivot_f_nb[self].load(cuda::memory_order_relaxed)) != 0) {
			/* f_and_not_b = nodeset->create_instance(0, (nodeset->pivot_f_nb[self].load(cuda::memory_order_relaxed)), 0, 0, false, 0, 0, 0) */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->f_and_not_b[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = nodeset->create_instance(0, (nodeset->pivot_f_nb[self].load(cuda::memory_order_relaxed)), 0, 0, false, 0, 0, 0);
				if (prev_val != new_val) {
					nodeset->f_and_not_b[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_f_nb = 0 */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->pivot_f_nb[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = 0;
				if (prev_val != new_val) {
					nodeset->pivot_f_nb[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

		}
		if ((nodeset->pivot_nf_b[self].load(cuda::memory_order_relaxed)) != 0) {
			/* not_f_and_b = nodeset->create_instance(0, 0, (nodeset->pivot_nf_b[self].load(cuda::memory_order_relaxed)), 0, false, 0, 0, 0) */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->not_f_and_b[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = nodeset->create_instance(0, 0, (nodeset->pivot_nf_b[self].load(cuda::memory_order_relaxed)), 0, false, 0, 0, 0);
				if (prev_val != new_val) {
					nodeset->not_f_and_b[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_b = 0 */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->pivot_nf_b[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = 0;
				if (prev_val != new_val) {
					nodeset->pivot_nf_b[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

		}
	}
	

	grid.sync();
	bool created_instances = false;
	if (t_idx == 0) {
		created_instances |= nodeset->sync_nrof_instances(host_nodeset);
		if (created_instances){
			FP->set();
		}
	}

}

__global__ void NodeSet_initialise_pivot_fwd_bwd(FPManager* FP,
												 inst_size nrof_instances,
												 NodeSet* const nodeset,
												 Node* const node,
												 Edge* const edge){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
	
		if (!(nodeset->scc[self].load(cuda::memory_order_relaxed))) {
			/* pivot_f_b.fwd = true */
			par_owner = (nodeset->pivot_f_b[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_f_b.bwd = true */
			par_owner = (nodeset->pivot_f_b[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_f_b = 0 */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->pivot_f_b[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = 0;
				if (prev_val != new_val) {
					nodeset->pivot_f_b[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_f_nb.fwd = true */
			par_owner = (nodeset->pivot_f_nb[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_f_nb.bwd = true */
			par_owner = (nodeset->pivot_f_nb[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_f_nb = 0 */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->pivot_f_nb[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = 0;
				if (prev_val != new_val) {
					nodeset->pivot_f_nb[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_b.fwd = true */
			par_owner = (nodeset->pivot_nf_b[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_b.bwd = true */
			par_owner = (nodeset->pivot_nf_b[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_b = 0 */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->pivot_nf_b[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = 0;
				if (prev_val != new_val) {
					nodeset->pivot_nf_b[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_nb.fwd = true */
			par_owner = (nodeset->pivot_nf_nb[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_nb.bwd = true */
			par_owner = (nodeset->pivot_nf_nb[self].load(cuda::memory_order_relaxed));
			if(par_owner != 0){
				ADL::BoolType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
				ADL::BoolType new_val = true;
				if (prev_val != new_val) {
					node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

			/* pivot_nf_nb = 0 */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = nodeset->pivot_nf_nb[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = 0;
				if (prev_val != new_val) {
					nodeset->pivot_nf_nb[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

		}
	}
	
}

__global__ void Node_pivots_nominate(FPManager* FP,
									 inst_size nrof_instances,
									 NodeSet* const nodeset,
									 Node* const node,
									 Edge* const edge){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
	
		if (!(nodeset->scc[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) {
			BoolType f = (node->fwd[self].load(cuda::memory_order_relaxed));
			BoolType b = (node->bwd[self].load(cuda::memory_order_relaxed));
			if ((f) && (b)) {
				/* set.pivot_f_b = self */
				par_owner = (node->set[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->pivot_f_b[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						nodeset->pivot_f_b[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if ((f) && (!(b))) {
				/* set.pivot_f_nb = self */
				par_owner = (node->set[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->pivot_f_nb[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						nodeset->pivot_f_nb[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if ((!(f)) && (b)) {
				/* set.pivot_nf_b = self */
				par_owner = (node->set[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->pivot_nf_b[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						nodeset->pivot_nf_b[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if ((!(f)) && (!(b))) {
				/* set.pivot_nf_nb = self */
				par_owner = (node->set[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::RefType prev_val = nodeset->pivot_nf_nb[par_owner].load(cuda::memory_order_relaxed);
					ADL::RefType new_val = self;
					if (prev_val != new_val) {
						nodeset->pivot_nf_nb[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
		}
	}
	
}

__global__ void Node_divide_into_sets_reset_fwd_bwd(FPManager* FP,
													inst_size nrof_instances,
													NodeSet* const nodeset,
													Node* const node,
													Edge* const edge){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
	
		BoolType f = (node->fwd[self].load(cuda::memory_order_relaxed));
		BoolType b = (node->bwd[self].load(cuda::memory_order_relaxed));
		if ((f) && (b)) {
			/* set = (nodeset->f_and_b[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = node->set[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = (nodeset->f_and_b[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
				if (prev_val != new_val) {
					node->set[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

		}
		if ((!(f)) && (b)) {
			/* set = (nodeset->not_f_and_b[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = node->set[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = (nodeset->not_f_and_b[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
				if (prev_val != new_val) {
					node->set[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

		}
		if ((f) && (!(b))) {
			/* set = (nodeset->f_and_not_b[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) */
			par_owner = self;
			if(par_owner != 0){
				ADL::RefType prev_val = node->set[par_owner].load(cuda::memory_order_relaxed);
				ADL::RefType new_val = (nodeset->f_and_not_b[node->set[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed));
				if (prev_val != new_val) {
					node->set[par_owner].store(new_val, cuda::memory_order_relaxed);
					FP->set();
				}
			}

		}
		/* fwd = false */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = false;
			if (prev_val != new_val) {
				node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

		/* bwd = false */
		par_owner = self;
		if(par_owner != 0){
			ADL::BoolType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
			ADL::BoolType new_val = false;
			if (prev_val != new_val) {
				node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
				FP->set();
			}
		}

	}
	
}

__global__ void Edge_compute_fwd_bwd(FPManager* FP,
									 inst_size nrof_instances,
									 NodeSet* const nodeset,
									 Node* const node,
									 Edge* const edge){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
	
		if ((node->set[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) == (node->set[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed))) {
			if (node->fwd[edge->s[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) {
				/* t.fwd = true */
				par_owner = (edge->t[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::BoolType prev_val = node->fwd[par_owner].load(cuda::memory_order_relaxed);
					ADL::BoolType new_val = true;
					if (prev_val != new_val) {
						node->fwd[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
			if (node->bwd[edge->t[self].load(cuda::memory_order_relaxed)].load(cuda::memory_order_relaxed)) {
				/* s.bwd = true */
				par_owner = (edge->s[self].load(cuda::memory_order_relaxed));
				if(par_owner != 0){
					ADL::BoolType prev_val = node->bwd[par_owner].load(cuda::memory_order_relaxed);
					ADL::BoolType new_val = true;
					if (prev_val != new_val) {
						node->bwd[par_owner].store(new_val, cuda::memory_order_relaxed);
						FP->set();
					}
				}

			}
		}
	}
	
}
__global__ void NodeSet_print(NodeSet* nodeset,
							  inst_size nrof_instances){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
		if (self != 0) {
			printf("NodeSet(%u): pivot_f_b=%u, pivot_f_nb=%u, pivot_nf_b=%u, pivot_nf_nb=%u, scc=%u, f_and_b=%u, not_f_and_b=%u, f_and_not_b=%u\n", self, nodeset->pivot_f_b[self].load(cuda::memory_order_relaxed), nodeset->pivot_f_nb[self].load(cuda::memory_order_relaxed), nodeset->pivot_nf_b[self].load(cuda::memory_order_relaxed), nodeset->pivot_nf_nb[self].load(cuda::memory_order_relaxed), nodeset->scc[self].load(cuda::memory_order_relaxed), nodeset->f_and_b[self].load(cuda::memory_order_relaxed), nodeset->not_f_and_b[self].load(cuda::memory_order_relaxed), nodeset->f_and_not_b[self].load(cuda::memory_order_relaxed));
		}
	}
}

__global__ void Node_print(Node* node,
						   inst_size nrof_instances){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
		if (self != 0) {
			printf("Node(%u): set=%u, fwd=%u, bwd=%u\n", self, node->set[self].load(cuda::memory_order_relaxed), node->fwd[self].load(cuda::memory_order_relaxed), node->bwd[self].load(cuda::memory_order_relaxed));
		}
	}
}

__global__ void Edge_print(Edge* edge,
						   inst_size nrof_instances){
	grid_group grid = this_grid();
	RefType t_idx = grid.thread_rank();
	inst_size num_threads = grid.size();
	RefType par_owner;
	for(RefType self = t_idx; self < nrof_instances; self += num_threads){
		if (self != 0) {
			printf("Edge(%u): s=%u, t=%u\n", self, edge->s[self].load(cuda::memory_order_relaxed), edge->t[self].load(cuda::memory_order_relaxed));
		}
	}
}


void launch_NodeSet_allocate_sets() {
	inst_size nrof_instances = host_NodeSet.nrof_instances();
	void* NodeSet_allocate_sets_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_NodeSet,
		&gm_Node,
		&gm_Edge,
		&host_NodeSet_ptr
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)NodeSet_allocate_sets);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)NodeSet_allocate_sets,
			std::get<0>(dims),
			std::get<1>(dims),
			NodeSet_allocate_sets_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_NodeSet_initialise_pivot_fwd_bwd() {
	inst_size nrof_instances = host_NodeSet.nrof_instances();
	void* NodeSet_initialise_pivot_fwd_bwd_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_NodeSet,
		&gm_Node,
		&gm_Edge
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)NodeSet_initialise_pivot_fwd_bwd);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)NodeSet_initialise_pivot_fwd_bwd,
			std::get<0>(dims),
			std::get<1>(dims),
			NodeSet_initialise_pivot_fwd_bwd_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Node_pivots_nominate() {
	inst_size nrof_instances = host_Node.nrof_instances();
	void* Node_pivots_nominate_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_NodeSet,
		&gm_Node,
		&gm_Edge
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Node_pivots_nominate);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Node_pivots_nominate,
			std::get<0>(dims),
			std::get<1>(dims),
			Node_pivots_nominate_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Node_divide_into_sets_reset_fwd_bwd() {
	inst_size nrof_instances = host_Node.nrof_instances();
	void* Node_divide_into_sets_reset_fwd_bwd_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_NodeSet,
		&gm_Node,
		&gm_Edge
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Node_divide_into_sets_reset_fwd_bwd);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Node_divide_into_sets_reset_fwd_bwd,
			std::get<0>(dims),
			std::get<1>(dims),
			Node_divide_into_sets_reset_fwd_bwd_args
		)
	);
	CHECK(cudaDeviceSynchronize());
}

void launch_Edge_compute_fwd_bwd() {
	inst_size nrof_instances = host_Edge.nrof_instances();
	void* Edge_compute_fwd_bwd_args[] = {
		&device_FP,
		&nrof_instances,
		&gm_NodeSet,
		&gm_Node,
		&gm_Edge
	};
	auto dims = ADL::get_launch_dims(nrof_instances, (void*)Edge_compute_fwd_bwd);

	CHECK(
		cudaLaunchCooperativeKernel(
			(void*)Edge_compute_fwd_bwd,
			std::get<0>(dims),
			std::get<1>(dims),
			Edge_compute_fwd_bwd_args
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
	CHECK(cudaHostRegister(&host_NodeSet, sizeof(NodeSet), cudaHostRegisterDefault));

	host_Edge.initialise(&structs[0], 100);
	host_Node.initialise(&structs[1], 100);
	host_NodeSet.initialise(&structs[2], 100);

	CHECK(cudaDeviceSynchronize());

	gm_Edge = (Edge*)host_Edge.to_device();
	gm_Node = (Node*)host_Node.to_device();
	gm_NodeSet = (NodeSet*)host_NodeSet.to_device();



	size_t printf_size;
	cudaDeviceGetLimit(&printf_size, cudaLimitPrintfFifoSize);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2 * printf_size);


	host_FP.push();
	device_FP = host_FP.to_device();



	host_FP.copy_from(device_FP);
	host_FP.push();
	do{
		host_FP.reset();
		host_FP.copy_to(device_FP);

		launch_Node_pivots_nominate();




		launch_NodeSet_allocate_sets();




		launch_Node_divide_into_sets_reset_fwd_bwd();




		launch_NodeSet_initialise_pivot_fwd_bwd();



		host_FP.copy_from(device_FP);
		host_FP.push();
		do{
			host_FP.reset();
			host_FP.copy_to(device_FP);

			launch_Edge_compute_fwd_bwd();



			host_FP.copy_from(device_FP);
			if(!host_FP.done()) host_FP.clear();
		}
		while(!host_FP.done());
		host_FP.pop();
		host_FP.copy_to(device_FP);

		host_FP.copy_from(device_FP);
		if(!host_FP.done()) host_FP.clear();
	}
	while(!host_FP.done());
	host_FP.pop();
	host_FP.copy_to(device_FP);

	Node_print<<<(host_Node.nrof_instances() + 512 - 1)/512, 512>>>(gm_Node, host_Node.nrof_instances());
	CHECK(cudaDeviceSynchronize());



	
}
