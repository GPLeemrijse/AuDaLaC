use crate::ast::Program;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;

pub enum DivisionStrategy {
	BlockSizeIncrease,
	GridSizeIncrease
}

pub struct WorkDivisor<'a> {
	program : &'a Program,
	instances_per_thread : usize,
	threads_per_block : usize,
	allocated_per_instance : usize,
	division_strategy : DivisionStrategy,
}

impl WorkDivisor<'_> {
	pub fn new(program : &Program, instances_per_thread : usize, threads_per_block : usize, allocated_per_instance : usize, 	division_strategy : DivisionStrategy,) -> WorkDivisor {
		WorkDivisor {
			program,
			instances_per_thread,
			threads_per_block,
			allocated_per_instance,
			division_strategy,
		}
	}

	pub fn get_dims(&self, kernel_name : &str) -> String {
		let nrof_instances = self.program.structs.len() * self.allocated_per_instance;
		let nrof_threads = (nrof_instances + self.instances_per_thread - 1) / self.instances_per_thread;
		formatdoc!("
			ADL::get_launch_dims({nrof_threads}, (void*){kernel_name})"
		)
	}


	fn loop_header(&self) -> String {
		use DivisionStrategy::*;

		match self.division_strategy {
			BlockSizeIncrease => {
				formatdoc!("
					\tfor(int i = 0; i < I_PER_THREAD; i++){{
					\t\tconst RefType self = block.size() * (i + grid.block_rank() * I_PER_THREAD) + block.thread_rank();
					\t\tif (self >= nrof_instances) break;"
				)
			},
			GridSizeIncrease => {
				formatdoc!("
					\tfor(int i = 0; i < I_PER_THREAD; i++){{
					\t\tconst RefType self = grid.thread_rank() + i * grid.size();
					\t\tif (self >= nrof_instances) break;"
				)
			},
		}
	}
}

impl CompileComponent for WorkDivisor<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<cooperative_groups.h>");
	}

	fn defines(&self) -> Option<String> {
		let ipt = self.instances_per_thread;
		let tpb = self.threads_per_block;

		Some(formatdoc!{"
			#define I_PER_THREAD {ipt}
			#define THREADS_PER_BLOCK {tpb}
		"})
	}
	fn typedefs(&self) -> Option<String> { None }

	fn globals(&self) -> Option<String> { None }
	
	fn functions(&self) -> Option<String> {
		let loop_header = self.loop_header();

		Some(formatdoc!("
			typedef void(*step_func)(RefType, bool*, uint64_t);
			template <step_func Step>
			__device__ void executeStep(inst_size nrof_instances, uint64_t struct_step_parity, grid_group grid, thread_block block, bool* stable){{
			{loop_header}

					Step(self, stable, struct_step_parity);
				}}
			}}
		"))
	}

	fn kernels(&self) -> Option<String> { None }
	
	fn pre_main(&self) -> Option<String> { None }
	
	fn main(&self) -> Option<String> { None }
	
	fn post_main(&self) -> Option<String> { None }
}