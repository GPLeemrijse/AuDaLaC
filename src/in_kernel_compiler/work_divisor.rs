use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;

pub enum DivisionStrategy {
	BlockSizeIncrease,
	GridSizeIncrease
}

pub struct WorkDivisor {
	instances_per_thread : usize,
	threads_per_block : usize,
	launched_threads : usize,
	division_strategy : DivisionStrategy,
}

impl WorkDivisor {
	pub fn new(instances_per_thread : usize, threads_per_block : usize, launched_threads : usize, 	division_strategy : DivisionStrategy,) -> WorkDivisor {
		WorkDivisor {
			instances_per_thread,
			threads_per_block,
			launched_threads,
			division_strategy,
		}
	}

	pub fn get_dims(&self, kernel_name : &str) -> String {
		let nrof_threads = self.launched_threads;
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
					\t\tconst RefType self = block.size() * (i + grid.block_rank() * I_PER_THREAD) + block.rank();
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

impl CompileComponent for WorkDivisor {
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
			template <typename Step>
			void executeStep(inst_size nrof_instances, bool step_parity, grid_group grid, thread_block block, bool* stable){{
			{loop_header}

					Step(self, stable, step_parity);
				}}
			}}
		"))
	}

	fn kernels(&self) -> Option<String> { None }
	
	fn pre_main(&self) -> Option<String> { None }
	
	fn main(&self) -> Option<String> { None }
	
	fn post_main(&self) -> Option<String> { None }
}