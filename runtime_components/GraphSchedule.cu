#include "GraphSchedule.h"
#include "ADL.h"

int GraphSchedule::Subgraph::nrof_subgraphs = 0;

void GraphSchedule::add_step(void* kernel, inst_size capacity, size_t smem) {
	int min_grid_size;
	int dyn_block_size;
	cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &dyn_block_size, kernel, 0, 0);

	if (capacity == 1) {
		dyn_block_size = 1;
	}
	dim3 blockDim(dyn_block_size);
	dim3 gridDim((capacity + dyn_block_size - 1) / dyn_block_size);

	k_params.func = kernel;
	k_params.gridDim = gridDim;
	k_params.blockDim = blockDim;
	k_params.sharedMemBytes = smem;
	void* args[1] = {
		(void*)&current->lvl
	};
	k_params.kernelParams = args;
	k_params.extra = NULL;

	cudaGraphNode_t new_node;
	if (current->last_node == NULL){
		CHECK(cudaGraphAddKernelNode(&new_node, current->graph, NULL, 0, &k_params));
	} else {
		CHECK(cudaGraphAddKernelNode(&new_node, current->graph, {&current->last_node}, 1, &k_params));
	}
	current->last_node = new_node;
}

void GraphSchedule::begin_fixpoint(void){
	current->next = new Subgraph(current->lvl + 1);
	fixpoints.push_back(current->next);
	current = current->next;
}

void GraphSchedule::end_fixpoint(void){
	current->fp_start = fixpoints.back();
	fixpoints.pop_back();
	current->next = new Subgraph(current->lvl - 1);
	current = current->next;
}

cudaGraphExec_t GraphSchedule::instantiate(cudaStream_t stream){
	cudaGraphExec_t result = head->instantiate(launch_kernel, relaunch_fp_kernel);
	head->fill_out_fixpoints(stream, relaunch_fp_kernel);
	return result;
}

void GraphSchedule::print_dot(void){
	head->print_dot();
}

void GraphSchedule::print_debug(void){
	head->print_debug();
}

GraphSchedule::GraphSchedule(void* launch_kernel, void* relaunch_fp_kernel) :
	launch_kernel(launch_kernel), relaunch_fp_kernel(relaunch_fp_kernel) {
	head = new Subgraph(-1);
	current = head;
}

GraphSchedule::Subgraph::Subgraph(int lvl) : lvl(lvl), next(NULL), last_node(NULL), fp_start(NULL) {
	CHECK(cudaGraphCreate(&graph, 0));
	number = ++Subgraph::nrof_subgraphs;
}

cudaGraphExec_t GraphSchedule::Subgraph::instantiate(void* launch_kernel, void* relaunch_fp_kernel) {
	cudaGraphExec_t next_to_launch;
	if(next == NULL) {
		next_to_launch = NULL;
	} else {
		next_to_launch = next->instantiate(launch_kernel, relaunch_fp_kernel);
	}

	// Holds kernel node parameters
	cudaKernelNodeParams k_params;
	
	// If we are the end of a fixpoint a conditional relaunch should be added
	if(fp_start != NULL) {
		k_params.func = relaunch_fp_kernel;
		k_params.gridDim = dim3(1, 1, 1);
		k_params.blockDim = dim3(1, 1, 1);
		k_params.sharedMemBytes = 0;
		
		void* n = NULL;
		void* args[3] = {
			(void*)&lvl,
			/*	At this point the start of our fixpoint is 
				not instantiated yet, we revisit this parameter later.*/
			(void*)&n, 
			(void*)&next_to_launch
		};
		k_params.kernelParams = args;
		k_params.extra = NULL;

		cudaGraphNode_t new_node;
		if (last_node == NULL){
			CHECK(cudaGraphAddKernelNode(&new_node, graph, NULL, 0, &k_params));
		} else {
			CHECK(cudaGraphAddKernelNode(&new_node, graph, &last_node, 1, &k_params));
		}
		last_node = new_node;
	} // Otherwise an unconditional launch suffices
	else if (next_to_launch != NULL) {
		// If we have nothing to execute, we directly execute the next step
		if (last_node == NULL){
			graph_exec = next_to_launch;
			return graph_exec;
		} else {
			cudaGraphNode_t new_node;
			k_params.func = launch_kernel;
			k_params.gridDim = dim3(1, 1, 1);
			k_params.blockDim = dim3(1, 1, 1);
			k_params.sharedMemBytes = 0;
			void* args[1] = {
				(void*)&next_to_launch
			};
			k_params.kernelParams = args;
			k_params.extra = NULL;
			CHECK(cudaGraphAddKernelNode(&new_node, graph, &last_node, 1, &k_params));
			last_node = new_node;
		}
	}

	// The graph is now complete, so we instantiate (if we have any nodes):
	if(last_node == NULL){
		graph_exec = NULL;
	} else {
		CHECK(cudaGraphInstantiate(&graph_exec, graph, cudaGraphInstantiateFlagDeviceLaunch));
	}
	return graph_exec;
}

void GraphSchedule::Subgraph::fill_out_fixpoints(cudaStream_t stream, void* relaunch_fp_kernel) {
	// If we have a fixpoint to start, we update our relaunch node.
	if(fp_start != NULL) {
		// Holds kernel node parameters
		cudaKernelNodeParams k_params;
		
		k_params.func = relaunch_fp_kernel;
		k_params.gridDim = dim3(1, 1, 1);
		k_params.blockDim = dim3(1, 1, 1);
		k_params.sharedMemBytes = 0;

		cudaGraphExec_t next_to_launch;
		if(next == NULL) {
			next_to_launch = NULL;
		} else {
			next_to_launch = next->graph_exec;
		}

		void* args[3] = {
			(void*)&lvl,
			(void*)&fp_start->graph_exec, 
			(void*)&next_to_launch
		};
		k_params.kernelParams = args;
		k_params.extra = NULL;

		CHECK(cudaGraphExecKernelNodeSetParams(graph_exec, last_node, &k_params));
	}
	/*	Now that our graph is complete, we upload it.
		We prevent uploading twice if we
		are directly launching next's graph.
	*/
	if((next == NULL || graph_exec != next->graph_exec) && graph_exec != NULL) {
		CHECK(cudaGraphUpload(graph_exec, stream));
	}

	if(next){
		next->fill_out_fixpoints(stream, relaunch_fp_kernel);
	}
}

void GraphSchedule::Subgraph::print_dot(){
	char dot_file[20];
	sprintf(dot_file, "graph%u.dot", number);
	CHECK(cudaGraphDebugDotPrint(graph, dot_file, cudaGraphDebugDotFlagsKernelNodeParams));
	if(next != NULL) {
		next->print_dot();
	}
}

void GraphSchedule::Subgraph::print_debug(){
	for (int i = 0; i < lvl + 1; i++){
		printf("\t");
	}
	printf("Subgraph %p(%u): graph=%p, graph_exec=%p, last_node=%p, fp_start=%p, next=%p\n", this, number, graph, graph_exec, last_node, fp_start, next);
	
	if(next != NULL) {
		next->print_debug();
	}
}