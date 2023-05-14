use crate::transpilation_traits::*;
use indoc::formatdoc;


pub struct NaiveFixpoint {

}

impl NaiveFixpoint {
	pub fn new() -> NaiveFixpoint {
		NaiveFixpoint {
		}
	}
}

impl FPStrategy for NaiveFixpoint {
	fn global_decl(&self) -> String {
		formatdoc!{"
			__device__ volatile bool fp_stack[FP_DEPTH];
			
			__device__ __inline__ void clear_stack(int lvl) {{
				while(lvl >= 0){{
					fp_stack[lvl--] = false;
				}}
			}}"
		}
	}

	fn is_stable(&self, lvl: usize) -> String {
		format!("fp_stack[{lvl}]")
	}

	fn set_unstable(&self, _: usize) -> String {
		format!("stable = false;")
	}

	fn top_of_kernel_decl(&self) -> String {
		String::new()
	}

	fn pre_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl+2);
		formatdoc!{"
			{indent}bool stable = true;
			{indent}if (is_thread0)
			{indent}	fp_stack[{lvl}] = true;
			{indent}grid.sync();
		"}
	}

	fn post_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl+2);
		format!{"{indent}if(!stable)\n{indent}	clear_stack({lvl});"}
	}
	
	fn initialise(&self) -> String {
		// No need to initialise, as that is done before each iteration (`pre_iteration`).
		"".to_string()
	}

	fn requires_intra_fixpoint_sync(&self) -> bool {
		false
	}
}