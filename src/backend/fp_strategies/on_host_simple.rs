use crate::backend::FPStrategy;
use indoc::formatdoc;

pub struct OnHostSimpleFixpoint {
	fp_depth: usize,
}

impl OnHostSimpleFixpoint {
	pub fn new(fp_depth: usize) -> OnHostSimpleFixpoint {
		OnHostSimpleFixpoint { fp_depth }
	}
}

impl FPStrategy for OnHostSimpleFixpoint {
	fn global_decl(&self) -> String {
		let fp_depth = self.fp_depth;
		let funcs = if fp_depth > 0 {
			formatdoc! {"
				__device__ bool fp_stack[FP_DEPTH];

				__device__ void clear_stack(int lvl) {{
					while(lvl >= 0){{
						fp_stack[lvl--] = false;
					}}
				}}
				__host__ bool load_fp_stack_from_host(int lvl) {{
					bool top_of_stack;
					CHECK(
						cudaMemcpyFromSymbol(
							&top_of_stack,
							fp_stack,
							sizeof(bool),
							sizeof(bool) * lvl
						)
					);
					return top_of_stack;
				}}

				__host__ void reset_fp_stack_from_host(int lvl) {{
					bool stack_entry = true;
					CHECK(
						cudaMemcpyToSymbol(
							fp_stack,
							&stack_entry,
							sizeof(bool),
							sizeof(bool) * lvl
						)
					);
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
