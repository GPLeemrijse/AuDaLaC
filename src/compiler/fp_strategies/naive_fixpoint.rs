use crate::compiler::FPStrategy;
use indoc::formatdoc;

pub struct NaiveFixpoint {
    fp_depth: usize,
}

impl NaiveFixpoint {
    pub fn new(fp_depth: usize) -> NaiveFixpoint {
        NaiveFixpoint { fp_depth }
    }
}

impl FPStrategy for NaiveFixpoint {
    fn global_decl(&self) -> String {
        let fp_depth = self.fp_depth;
        let stack_and_clear = if fp_depth > 0 {
            formatdoc! {"
				__device__ cuda::atomic<bool, cuda::thread_scope_device> fp_stack[FP_DEPTH];

				__device__ void clear_stack(int lvl) {{
					while(lvl >= 0){{
						fp_stack[lvl--].store(false, cuda::memory_order_relaxed);
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
        format!("fp_stack[{lvl}].load(cuda::memory_order_relaxed)")
    }

    fn set_unstable(&self, _: usize) -> String {
        format!("stable = false;")
    }

    fn top_of_kernel_decl(&self) -> String {
        String::new()
    }

    fn pre_iteration(&self, lvl: usize) -> String {
        let indent = "\t".repeat(lvl + 2);
        formatdoc! {"
			{indent}bool stable = true;
			{indent}grid.sync();
			{indent}if (is_thread0)
			{indent}	fp_stack[{lvl}].store(true, cuda::memory_order_relaxed);
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
}
