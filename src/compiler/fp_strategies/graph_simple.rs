use crate::compiler::FPStrategy;
use indoc::formatdoc;

pub struct GraphSimpleFixpoint {
	fp_depth: usize,
}

impl GraphSimpleFixpoint {
	pub fn new(fp_depth: usize) -> GraphSimpleFixpoint {
		GraphSimpleFixpoint { fp_depth }
	}
}

impl FPStrategy for GraphSimpleFixpoint {
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
			\tbool stable = true;
			\tbool* stable_ptr = &stable;"
		}
	}

	fn post_kernel(&self) -> String {
		format! {"\n\tif(!stable && fp_lvl >= 0)\n\t\tclear_stack(fp_lvl);"}
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
