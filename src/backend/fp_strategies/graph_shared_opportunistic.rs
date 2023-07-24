use crate::backend::FPStrategy;
use indoc::formatdoc;

pub struct GraphSharedBanksOpportunisticFixpoint {
	fp_depth: usize,
}

impl GraphSharedBanksOpportunisticFixpoint {
	pub fn new(fp_depth: usize) -> GraphSharedBanksOpportunisticFixpoint {
		GraphSharedBanksOpportunisticFixpoint { fp_depth }
	}
}

impl FPStrategy for GraphSharedBanksOpportunisticFixpoint {
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
			\t__shared__ uint32_t stable[32];
			\tif(bl_rank < 32){{
			\t	stable[bl_rank] = (uint32_t)true;
			\t}}
			\tbool* stable_ptr = (bool*)&stable[bl_rank % 32];
			\tif(fp_lvl >= 0) __syncthreads();
		"}
	}

	fn post_kernel(&self) -> String {
		formatdoc!{"
			\tif(fp_lvl >= 0){{
			\t\tbool stable_reduced = __all_sync(__activemask(), stable[bl_rank % 32]);
			\t\tif(!stable_reduced){{
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
