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
			__device__ cuda::atomic<bool, cuda::thread_scope_device> fp_stack[FP_DEPTH][2];
			
			__device__ __inline__ void clear_stack(int lvl, bool* iteration_parity) {{
				/*	Clears the stack on the iteration_parity side.
					The !iteration_parity side is currently (being set to) true.
				*/
				while(lvl >= 0){{
					fp_stack[lvl][(uint)iteration_parity[lvl]].store(false, cuda::memory_order_relaxed);
					lvl--;
				}}
			}}"
		}
	}

	fn is_stable(&self, lvl: usize) -> String {
		format!("fp_stack[{lvl}][(uint)iteration_parity[{lvl}]].load(cuda::memory_order_relaxed)")
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
			{indent}if (is_thread0)
			{indent}	fp_stack[{lvl}][(uint)!iteration_parity[{lvl}]].store(true, cuda::memory_order_relaxed);
		"}
	}

	fn post_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl+2);
		formatdoc!{"
			{indent}if(!stable)
			{indent}	clear_stack({lvl}, iteration_parity[{lvl}]);"
		}
	}

	fn initialise(&self) -> String {
		formatdoc!("
			\tcuda::atomic<bool, cuda::thread_scope_device>* fp_stack_address;
			\tcudaGetSymbolAddress((void **)&fp_stack_address, fp_stack);
			\tCHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 2 * sizeof(cuda::atomic<bool, cuda::thread_scope_device>)));"
		)
	}

	fn requires_intra_fixpoint_sync(&self) -> bool {
		true
	}
}