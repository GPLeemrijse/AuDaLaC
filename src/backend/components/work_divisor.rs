use crate::backend::compilation_traits::*;
use indoc::formatdoc;
use std::collections::BTreeSet;

pub enum DivisionStrategy {
	Dynamic,
}

pub struct WorkDivisor {
	division_strategy: DivisionStrategy,
	print_unstable: bool
}

impl<'a> WorkDivisor {
	pub fn new(
		division_strategy: DivisionStrategy,
		print_unstable: bool
	) -> WorkDivisor {
		WorkDivisor {
			division_strategy,
			print_unstable
		}
	}

	pub fn get_dims(&self, kernel_name: &str, print: bool) -> String {
		formatdoc!(
			"
			get_launch_dims(max_nrof_executing_instances, (void*){kernel_name}, {print})",
		)
	}

	pub fn execute_step(&self, kernel_name: &String, nrof_instances: String) -> String {
		if self.print_unstable {
			formatdoc!("executeStep<{kernel_name}>({nrof_instances}, grid, block, &stable, \"{kernel_name}\");")
		} else {
			formatdoc!("executeStep<{kernel_name}>({nrof_instances}, grid, block, &stable);")
		}
	}

	fn loop_header(&self) -> String {
		use DivisionStrategy::*;

		match self.division_strategy {
			Dynamic => {
				formatdoc!(
					"
					\tfor(RefType self = grid.thread_rank(); self < nrof_instances; self += grid.size()){{"
				)
			}
		}
	}

	fn launch_dims_function(&self) -> String {
		formatdoc! {"
			__host__ std::tuple<dim3, dim3> get_launch_dims(inst_size max_nrof_executing_instances, const void* kernel, bool print = false){{
				int min_grid_size;
				int dyn_block_size;
				cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &dyn_block_size, kernel, 0, 0);

				int numBlocksPerSm = 0;
				cudaDeviceProp deviceProp;
				cudaGetDeviceProperties(&deviceProp, 0);
				cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, dyn_block_size, 0);
			  
				int max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;
				int wanted_blocks = (max_nrof_executing_instances + dyn_block_size - 1)/dyn_block_size;
				int used_blocks = min(max_blocks, wanted_blocks);
				int nrof_threads = used_blocks * dyn_block_size;

				if (used_blocks == 0) {{
					fprintf(stderr, \"Could not fit kernel on device!\\n\");
					exit(1234);
				}}

				if (print) {{
					fprintf(stderr, \"A maximum of %u instances will execute.\\n\", max_nrof_executing_instances);
					fprintf(stderr, \"Launching %u/%u blocks of %u threads = %u threads.\\n\", used_blocks, max_blocks, dyn_block_size, nrof_threads);
					fprintf(stderr, \"Resulting in max %u instances per thread.\\n\", (max_nrof_executing_instances + nrof_threads - 1) / nrof_threads);
				}}

				dim3 dimBlock(dyn_block_size, 1, 1);
				dim3 dimGrid(used_blocks, 1, 1);
				return std::make_tuple(dimGrid, dimBlock);
			}}

		"}
	}

	fn execute_step_function(&self) -> String {
		let loop_header = self.loop_header();
		let execution = if self.print_unstable {
			formatdoc!(
				"
				\t\tbool __stable = true;
				\t\tStep(self, &__stable);
				\t\tif (!__stable) {{
				\t\t\tprintf(\"Step %s was unstable (instance %u)\\n\", step_str, self);
				\t\t\t*stable = false;
				\t\t}}"
			)
		} else {
			formatdoc!(
				"
				\t\tStep(self, stable);"
			)
		};

		let step_str_par = if self.print_unstable {
			", const char* step_str"
		} else {
			""
		};

		formatdoc!("
			typedef void(*step_func)(RefType, bool*);
			template <step_func Step>
			__device__ void executeStep(inst_size nrof_instances, grid_group grid, thread_block block, bool* stable{step_str_par}){{
			{loop_header}

			{execution}
				}}
			}}

		")
	}
}

impl CompileComponent for WorkDivisor {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<cooperative_groups.h>");
		set.insert("<tuple>");
	}

	fn defines(&self) -> Option<String> {
		None
	}
	fn typedefs(&self) -> Option<String> {
		None
	}

	fn globals(&self) -> Option<String> {
		None
	}

	fn functions(&self) -> Option<String> {
		Some(self.execute_step_function() + &self.launch_dims_function())
	}

	fn kernels(&self) -> Option<String> {
		None
	}

	fn pre_main(&self) -> Option<String> {
		None
	}

	fn main(&self) -> Option<String> {
		None
	}

	fn post_main(&self) -> Option<String> {
		None
	}
}
