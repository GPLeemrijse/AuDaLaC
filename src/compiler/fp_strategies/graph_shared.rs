use crate::compiler::FPStrategy;
use indoc::formatdoc;

pub struct GraphSharedFixpoint {
	fp_depth: usize,
}

impl GraphSharedFixpoint {
	pub fn new(fp_depth: usize) -> GraphSharedFixpoint {
		GraphSharedFixpoint { fp_depth }
	}
}

impl FPStrategy for GraphSharedFixpoint {
	fn global_decl(&self) -> String {
		let fp_depth = self.fp_depth;
		let funcs = if fp_depth > 0 {
			formatdoc! {"
				__device__ volatile bool fp_stack[FP_DEPTH];

				__device__ void clear_stack(int lvl){{
					while(lvl >= 0){{
						fp_stack[lvl--] = false;
					}}
				}}
				"
			}
		} else {
			"".to_string()
		};
		formatdoc! {"
			#define FP_DEPTH {fp_depth}
			{funcs}"
		}
	}

	fn is_stable(&self, _: usize) -> String {
		format!{"fp_stack[fp_lvl]"}
	}

	fn top_of_kernel_decl(&self) -> String {
		formatdoc!{"
			\tconst uint bl_rank = this_thread_block().thread_rank();
			\t__shared__ bool stable;
			\tif(bl_rank == 0){{
			\t	stable = true;
			\t}}
			\tbool* stable_ptr = &stable;
			\tif(fp_lvl >= 0) __syncthreads();
		"}
	}

	fn post_kernel(&self) -> String {
		formatdoc!{"
			\tif(fp_lvl >= 0){{
			\t\t__syncthreads();
			\t\tif(bl_rank == 0 && !stable){{
			\t\t\tclear_stack(fp_lvl);
			\t\t}}
			\t}}
		"}
	}

	fn pre_iteration(&self, _: usize) -> String {
		String::new()
	}

	fn post_iteration(&self, _: usize) -> String {
		String::new()
	}

	fn initialise(&self) -> String {
		String::new()
	}

	fn requires_intra_fixpoint_sync(&self) -> bool {
		false
	}
}
