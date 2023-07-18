#ifndef SCHEDULE_H
#define SCHEDULE_H
#include <vector>
#include "ADL.h"
using namespace ADL;

class Schedule {
	class Subgraph {
		static int nrof_subgraphs;
	public:
		int lvl;
		Subgraph* next;
		cudaGraph_t graph;
		cudaGraphExec_t graph_exec;
		cudaGraphNode_t last_node;
		int number;
		
		Subgraph* fp_start;
		
		Subgraph(int lvl);

		cudaGraphExec_t instantiate(void* launch_kernel, void* relaunch_fp_kernel);
		void fill_out_fixpoints(cudaStream_t stream, void* relaunch_fp_kernel);

		void print_dot(void);

		void print_debug(void);

	};

	void* launch_kernel;
	void* relaunch_fp_kernel;
	cudaKernelNodeParams k_params;
	Schedule::Subgraph* head;
	Schedule::Subgraph* current;
	std::vector<Schedule::Subgraph*> fixpoints;


public:
	Schedule(void* launch_kernel, void* relaunch_fp_kernel);

	void add_step(void* kernel, inst_size capacity, size_t smem);

	void begin_fixpoint(void);
	void end_fixpoint(void);

	cudaGraphExec_t instantiate(cudaStream_t stream);

	void print_dot(void);

	void print_debug(void);
};

#endif