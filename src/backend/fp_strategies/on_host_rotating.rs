use crate::backend::FPStrategy;
use indoc::formatdoc;

pub struct OnHostRotatingFixpoint {
	fp_depth: usize,
}

impl OnHostRotatingFixpoint {
	pub fn new(fp_depth: usize) -> OnHostRotatingFixpoint {
		OnHostRotatingFixpoint { fp_depth }
	}
}

impl FPStrategy for OnHostRotatingFixpoint {
	fn global_decl(&self) -> String {
		let fp_depth = self.fp_depth;
		let funcs = if fp_depth > 0 {
			formatdoc! {"
				/* Transform an iter_idx into the fp_stack index
				   associated with that operation.
				*/
				#define FP_SET(X) (X)
				#define FP_RESET(X) ((X) + 1 >= 3 ? (X) + 1 - 3 : (X) + 1)
				#define FP_READ(X) ((X) + 2 >= 3 ? (X) + 2 - 3 : (X) + 2)

				__device__ bool fp_stack[FP_DEPTH][3];

				typedef struct {{
					uint8_t idxs[FP_DEPTH];
				}} iter_idx_t;

				__device__ void clear_stack(int lvl, iter_idx_t iter_idx) {{
					/*	Clears the stack on the FP_SET side.
						The FP_RESET and FP_READ sides should remain the same.
					*/
					while(lvl >= 0){{
						fp_stack[lvl][FP_SET(iter_idx.idxs[lvl])] = false;
						lvl--;
					}}
				}}

				__host__ bool load_fp_stack_from_host(int lvl, iter_idx_t iter_idx, cudaStream_t kernel_stream, cudaStream_t reset_fp_stream) {{
					CHECK(cudaStreamSynchronize(reset_fp_stream));
					bool top_of_stack;
					CHECK(
						cudaMemcpyFromSymbolAsync(
							&top_of_stack,
							fp_stack,
							sizeof(bool),
							sizeof(bool) * (3 * lvl + FP_READ(iter_idx.idxs[lvl])),
							cudaMemcpyDefault,
							kernel_stream
						)
					);
					return top_of_stack;
				}}

				__host__ void reset_fp_stack_from_host(int lvl, iter_idx_t iter_idx, cudaStream_t stream) {{
					bool stack_entry = true;
					CHECK(
						cudaMemcpyToSymbolAsync(
							fp_stack,
							&stack_entry,
							sizeof(bool),
							sizeof(bool) * (3 * lvl + FP_RESET(iter_idx.idxs[lvl])),
							cudaMemcpyDefault,
							stream 
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
		format!{"load_fp_stack_from_host({lvl}, iter_idx, kernel_stream, reset_fp_stream)"}
	}

	fn top_of_kernel_decl(&self) -> String {
		format!("\n\tbool stable = true;")
	}

	fn post_kernel(&self) -> String {
		format! {"\n\tif(!stable && fp_lvl >= 0)\n\t\tclear_stack(fp_lvl, iter_idx);"}
	}

	fn pre_iteration(&self, _: usize) -> String {
		// let indent = "\t".repeat(lvl + 2);
		// formatdoc!{"
		// 	{indent}reset_fp_stack_from_host({lvl}, iter_idx, reset_fp_stream);
		// "}
		String::new()
	}

	fn post_iteration(&self, lvl: usize) -> String {
		let indent = "\t".repeat(lvl + 2);
		formatdoc!{"
			{indent}reset_fp_stack_from_host({lvl}, iter_idx, reset_fp_stream);
			{indent}/* The next index to set is the one that has been reset. */
			{indent}iter_idx.idxs[{lvl}] = FP_RESET(iter_idx.idxs[{lvl}]);"
		}
	}

	fn initialise(&self) -> String {
		let mut result = formatdoc!{
			"\titer_idx_t iter_idx = {{0}};
			\tcudaStream_t reset_fp_stream;
			\tcudaStreamCreate(&reset_fp_stream);
			"
		};

		if self.fp_depth > 0 {
            result.push_str(&formatdoc!("
				\tbool* fp_stack_address;
				\tCHECK(cudaGetSymbolAddress((void **)&fp_stack_address, fp_stack));
				\tCHECK(cudaMemset((void*)fp_stack_address, 1, FP_DEPTH * 3 * sizeof(bool)));"
			));
        }
        result
	}

	fn requires_intra_fixpoint_sync(&self) -> bool {
		false
	}
}