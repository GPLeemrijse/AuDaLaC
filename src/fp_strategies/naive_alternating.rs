use crate::transpilation_traits::*;
use indoc::formatdoc;


pub struct NaiveAlternatingFixpoint {

}

impl NaiveAlternatingFixpoint {
	pub fn new() -> NaiveAlternatingFixpoint {
		NaiveAlternatingFixpoint {
		}
	}
}

impl FPStrategy for NaiveAlternatingFixpoint {
	fn global_decl(&self) -> String {
		formatdoc!{"
			__device__ bool fp_stack[FP_DEPTH][2];
			
			__device__ __inline__ void clear_stack(int lvl, bool iteration_parity) {{
				/*	For the first lvl, only clear the iteration_parity bool.
					The first !iteration_parity bool is being set to true in advance.
				*/
				fp_stack[lvl--][iteration_parity] = false;

				while(lvl >= 0){{
					fp_stack[lvl][iteration_parity] = false;
					fp_stack[lvl][!iteration_parity] = false;
					lvl--;
				}}
			}}"
		}
	}

	fn is_stable(&self, lvl: usize) -> String {
		format!("fp_stack[{lvl}][iteration_parity[{lvl}]]")
	}

	fn set_unstable(&self, _: usize) -> String {
		format!("stable = false;")
	}

	fn top_of_kernel_decl(&self) -> String {
		formatdoc!("
			\tbool iteration_parity[FP_DEPTH] = {{false}};"
		)
	}

	fn pre_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl+2);
		formatdoc!{"
			{indent}iteration_parity[{lvl}] = !iteration_parity[{lvl}];
			{indent}bool stable = true;
			{indent}if (in_grid_rank == 0)
			{indent}	fp_stack[{lvl}][!iteration_parity[{lvl}]] = true;
		"}
	}

	fn post_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl+2);
		format!{"{indent}if(!stable)\n{indent}	clear_stack({lvl}, iteration_parity[{lvl}]);"}
	}

	fn pre_step_function(&self, _: usize) -> String {
		"bool stable = true;".to_string()
	}

	fn post_step_function(&self, _: usize) -> String {
		"return stable;".to_string()
	}

	fn initialise(&self) -> String {
		formatdoc!("
			\tbool* fp_stack_address;
			\tcudaGetSymbolAddress((void **)&fp_stack_address, fp_stack);
			\tCHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 2 * sizeof(bool)));"
		)
	}

	fn requires_intra_fixpoint_sync(&self) -> bool {
		true
	}
}