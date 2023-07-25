use crate::compiler::FPStrategy;
use indoc::formatdoc;

pub struct InKernelSimpleFixpoint {
    fp_depth: usize,
}

impl InKernelSimpleFixpoint {
    pub fn new(fp_depth: usize) -> InKernelSimpleFixpoint {
        InKernelSimpleFixpoint { fp_depth }
    }
}

impl FPStrategy for InKernelSimpleFixpoint {
    fn global_decl(&self) -> String {
        let fp_depth = self.fp_depth;
        let stack_and_clear = if fp_depth > 0 {
            formatdoc! {"
				__device__ bool fp_stack[FP_DEPTH];

				__device__ void clear_stack(int lvl) {{
					while(lvl >= 0){{
						fp_stack[lvl--] = false;
					}}
				}}"
            }
        } else {
            "".to_string()
        };
        formatdoc! {"
			#define FP_DEPTH {fp_depth}
			{stack_and_clear}"
        }
    }

    fn is_stable(&self, lvl: usize) -> String {
        format!("fp_stack[{lvl}]")
    }

    fn top_of_kernel_decl(&self) -> String {
        String::new()
    }

    fn pre_iteration(&self, lvl: usize) -> String {
        let indent = "\t".repeat(lvl + 2);
        formatdoc! {"
			{indent}bool stable = true;
			{indent}grid.sync();
			{indent}if (grid.thread_rank() == 0)
			{indent}	fp_stack[{lvl}] = true;
			{indent}grid.sync();
		"}
    }

    fn post_iteration(&self, lvl: usize) -> String {
        let indent = "\t".repeat(lvl + 2);
        format! {"{indent}if(!stable)\n{indent}	clear_stack({lvl});"}
    }

    fn initialise(&self) -> String {
        // No need to initialise, as that is done before each iteration (`pre_iteration`).
        "".to_string()
    }

    fn requires_intra_fixpoint_sync(&self) -> bool {
        false
    }

    fn post_kernel(&self) -> String {
        String::new()
    }
}
