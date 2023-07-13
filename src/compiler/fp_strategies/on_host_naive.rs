use crate::compiler::FPStrategy;
use indoc::formatdoc;

pub struct OnHostNaiveFixpoint {
	fp_depth: usize,
}

impl OnHostNaiveFixpoint {
	pub fn new(fp_depth: usize) -> OnHostNaiveFixpoint {
		OnHostNaiveFixpoint { fp_depth }
	}
}

impl FPStrategy for OnHostNaiveFixpoint {
	fn global_decl(&self) -> String {
		let fp_depth = self.fp_depth;
		let funcs = if fp_depth > 0 {
			formatdoc! {"
				__device__ cuda::atomic<bool, cuda::thread_scope_system> fp_stack[FP_DEPTH];

				__device__ void clear_stack(int lvl) {{
					while(lvl >= 0){{
						fp_stack[lvl--].store(false, cuda::memory_order_relaxed);
					}}
				}}
				__host__ bool load_fp_stack_from_host(int lvl) {{
                    CHECK(cudaDeviceSynchronize());
					cuda::atomic<bool, cuda::thread_scope_device> top_of_stack;
					CHECK(
						cudaMemcpyFromSymbol(
							&top_of_stack,
							fp_stack,
							sizeof(cuda::atomic<bool, cuda::thread_scope_device>),
							sizeof(cuda::atomic<bool, cuda::thread_scope_device>) * lvl
						)
					);
					return top_of_stack.load(cuda::memory_order_relaxed);
				}}

				__host__ void reset_fp_stack_from_host(int lvl) {{
					cuda::atomic<bool, cuda::thread_scope_device> stack_entry(true);
					CHECK(
						cudaMemcpyToSymbol(
							fp_stack,
							&stack_entry,
							sizeof(cuda::atomic<bool, cuda::thread_scope_device>),
							sizeof(cuda::atomic<bool, cuda::thread_scope_device>) * lvl
						)
					);
                    CHECK(cudaDeviceSynchronize());
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

	/* Can only be called from Host */
	fn is_stable(&self, lvl: usize) -> String {
		format!{"load_fp_stack_from_host({lvl})"}
	}

	fn set_unstable(&self, _: usize) -> String {
		format!("stable = false;")
	}

	fn top_of_kernel_decl(&self) -> String {
		format!("\n\tbool stable = true;")
	}

	fn post_kernel(&self) -> String {
		format! {"\n\tif(!stable && fp_lvl >= 0)\n\t\tclear_stack(fp_lvl);"}
	}

	fn pre_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl + 2);
		format!("\n{indent}reset_fp_stack_from_host({lvl});")
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
