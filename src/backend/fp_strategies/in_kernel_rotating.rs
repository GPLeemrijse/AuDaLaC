use crate::backend::FPStrategy;
use indoc::formatdoc;

pub struct InKernelRotatingFixpoint {
    fp_depth: usize,
}

impl InKernelRotatingFixpoint {
    pub fn new(fp_depth: usize) -> InKernelRotatingFixpoint {
        InKernelRotatingFixpoint { fp_depth }
    }
}

impl FPStrategy for InKernelRotatingFixpoint {
    fn global_decl(&self) -> String {
        let fp_depth = self.fp_depth;
        let stack_and_clear = if fp_depth > 0 {
            formatdoc! {"
				/* Transform an iter_idx into the fp_stack index
				   associated with that operation.
				*/
				#define FP_SET(X) (X)
				#define FP_RESET(X) ((X) + 1 >= 3 ? (X) + 1 - 3 : (X) + 1)
				#define FP_READ(X) ((X) + 2 >= 3 ? (X) + 2 - 3 : (X) + 2)

				__device__ bool fp_stack[FP_DEPTH][3];

				__device__ void clear_stack(int lvl, uint8_t* iter_idx) {{
					/*	Clears the stack on the FP_SET side.
						The FP_RESET and FP_READ sides should remain the same.
					*/
					while(lvl >= 0){{
						fp_stack[lvl][FP_SET(iter_idx[lvl])] = false;
						lvl--;
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
        format!("fp_stack[{lvl}][FP_READ(iter_idx[{lvl}])]")
    }

    fn top_of_kernel_decl(&self) -> String {
        if self.fp_depth > 0 {
            formatdoc!("
				\tuint8_t iter_idx[FP_DEPTH] = {{0}}; // Denotes which fp_stack index ([0, 2]) is currently being set."
			)
        } else {
            "".to_string()
        }
    }

    fn pre_iteration(&self, lvl: usize) -> String {
        let indent = "\t".repeat(lvl + 2);
        formatdoc! {"
			{indent}bool stable = true;
			{indent}if (grid.thread_rank() == 0){{
			{indent}	/* Resets the next fp_stack index in advance. */
			{indent}	fp_stack[{lvl}][FP_RESET(iter_idx[{lvl}])] = true;
			{indent}}}
		"}
    }

    fn post_iteration(&self, lvl: usize) -> String {
        let indent = "\t".repeat(lvl + 2);
        formatdoc! {"
			{indent}if(!stable){{
			{indent}	clear_stack({lvl}, iter_idx);
			{indent}}}
			{indent}/* The next index to set is the one that has been reset. */
			{indent}iter_idx[{lvl}] = FP_RESET(iter_idx[{lvl}]);"
        }
    }

    fn initialise(&self) -> String {
        if self.fp_depth > 0 {
            formatdoc!("
				\tbool* fp_stack_address;
				\tCHECK(cudaGetSymbolAddress((void **)&fp_stack_address, fp_stack));
				\tCHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 3 * sizeof(bool)));"
			)
        } else {
            "".to_string()
        }
    }

    fn requires_intra_fixpoint_sync(&self) -> bool {
        true
    }

    fn post_kernel(&self) -> String {
        String::new()
    }
}
