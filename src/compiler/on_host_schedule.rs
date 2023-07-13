use crate::analysis::constructors;
use crate::utils::format_signature;
use crate::analysis::get_step_to_structs;
use crate::compiler::compilation_traits::*;
use crate::compiler::components::WorkDivisor;
use crate::compiler::StepBodyCompiler;
use crate::parser::ast::*;
use indoc::formatdoc;
use std::collections::BTreeSet;
use std::collections::HashMap;

pub struct OnHostSchedule<'a> {
	program: &'a Program,
	fp: &'a dyn FPStrategy,
	step_to_structs: HashMap<&'a String, Vec<&'a ADLStruct>>,
	step_transpiler: &'a StepBodyCompiler<'a>,
	work_divisor: &'a WorkDivisor,
}

impl OnHostSchedule<'_> {
	pub fn new<'a>(
		program: &'a Program,
		fp: &'a dyn FPStrategy,
		step_transpiler: &'a StepBodyCompiler<'_>,
		work_divisor: &'a WorkDivisor,
	) -> OnHostSchedule<'a> {
		OnHostSchedule {
			program,
			fp,
			step_to_structs: get_step_to_structs(program),
			step_transpiler: step_transpiler,
			work_divisor,
		}
	}

	fn kernel_name(&self, strct: &String, step: &String) -> String {
		format!("kernel_{strct}_{step}")
	}

	fn copy_counters_from_struct_as_c(&self) -> String {
		formatdoc!{"
			__host__ inst_size get_created_instances_from(void* struct_ptr) {{
				cuda::atomic<inst_size, cuda::thread_scope_device> dest;
				CHECK(
					cudaMemcpy(
						&dest,
						(void*)((size_t)struct_ptr + created_instances_offset),
						sizeof(cuda::atomic<inst_size, cuda::thread_scope_device>),
						cudaMemcpyDeviceToHost
					)
				);
				return dest.load(cuda::memory_order_relaxed);
			}}
		"}
	}

	fn schedule_as_c(&self, sched : &Schedule, fp_level: usize) -> String {
		use Schedule::*;
		match sched {
			StepCall(..) => self.stepcall_as_c(sched, fp_level),
			TypedStepCall(..) => self.typedstepcall_as_c(sched, fp_level),
			Sequential(..) => self.sequential_as_c(sched, fp_level),
			Fixpoint(..) => self.fixpoint_as_c(sched, fp_level),
		}
	}

	fn stepcall_as_c(&self, stepcall : &Schedule, fp_level: usize) -> String {
		use Schedule::*;
		if let StepCall(step_name, _) = stepcall {
			let mut structs = self.step_to_structs.get(step_name).unwrap().iter();

			let mut combined_sched = TypedStepCall(structs.next()
														  .expect("Need at least one implementor")
														  .name
														  .clone(),
												   step_name.to_string(),
												   (0, 0));

			// Transform into sequence of typed step calls
			for s in structs {
				combined_sched = Sequential(
					Box::new(TypedStepCall(s.name.clone(), step_name.to_string(), (0, 0))),
					Box::new(combined_sched),
					(0, 0)
				);
			}

			self.schedule_as_c(&combined_sched, fp_level)
		} else {
			unreachable!()
		}
	}


	fn typedstepcall_as_c(&self, typedstepcall : &Schedule, fp_level: usize) -> String {
		use Schedule::*;
		let indent = "\t".repeat(fp_level + 1);

		if let TypedStepCall(strct_name, step_name, _) = typedstepcall {
			let step = self.program.step_by_name(strct_name, step_name).unwrap();

			let constructors = constructors(step);
			let post_launch = if !constructors.is_empty() {
				constructors.iter()
							.map(|strct|
								format!(
									"\n{indent}nrof_{strct}s = get_created_instances_from(loc_{});",
									strct.to_lowercase()
								)
							)
							.collect::<Vec<String>>()
							.join("\n")
			} else {
				String::new()
			};

			formatdoc!{"
				{indent}{strct_name}_{step_name}.launch();{post_launch}
				"
			}
		} else {
			unreachable!();
		}
	}

	fn sequential_as_c(&self, sequential : &Schedule, fp_level: usize) -> String {
		use Schedule::*;
		if let Sequential(s1, s2, _) = sequential {
			let s1_as_c = self.schedule_as_c(s1, fp_level);
			let s2_as_c = self.schedule_as_c(s2, fp_level);
			formatdoc!("
				{s1_as_c}
				{s2_as_c}"
			)
		} else {
			unreachable!();
		}
	}

	fn fixpoint_as_c(&self, fixpoint : &Schedule, fp_level: usize) -> String {
		use Schedule::*;
		if let Fixpoint(s, _) = fixpoint {
			let indent = "\t".repeat(fp_level + 1);
			let pre_iteration = self.fp.pre_iteration(fp_level);
			let post_iteration = self.fp.post_iteration(fp_level);
			let sched = self.schedule_as_c(s, fp_level + 1);
			let is_stable = self.fp.is_stable(fp_level);

			formatdoc! {"
				{indent}fp_lvl++;
				{indent}do{{{pre_iteration}
				{sched}{post_iteration}
				{indent}}} while(!{is_stable});
				{indent}fp_lvl--;
			"}
		} else {
			unreachable!();
		}
	}

	fn step_as_kernel(&self, strct: &String, step: &String) -> String {
		let kernel_name = self.kernel_name(strct, step);
		let func_header = format!("__global__ __launch_bounds__(THREADS_PER_BLOCK) void {kernel_name}");

		let params = vec!["inst_size nrof_instances".to_string(), "int fp_lvl".to_string()];

		let kernel_signature = format_signature(&func_header, &params, 0);

		let execute_step = self.work_divisor
							   .execute_step(
									&self.step_transpiler.execute_f_name(strct, step),
									"nrof_instances".to_string()
							   );
		let pre = self.fp.top_of_kernel_decl();
		let post = self.fp.post_kernel();

		formatdoc! {"
			{kernel_signature}
				const grid_group grid = this_grid();
				const thread_block block = this_thread_block();{pre}
				{execute_step}{post}
			}}
		"}
	}
}

impl CompileComponent for OnHostSchedule<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<cooperative_groups.h>");
	}

	fn defines(&self) -> Option<String> {
		None
	}

	fn typedefs(&self) -> Option<String> {
		let cg = "using namespace cooperative_groups;".to_string();
		let launch_params = formatdoc!{"
			class Step {{
				void* kernel;
				void* args[2];
				dim3 gridDim;
				dim3 blockDim;
				uint max_blocks;
				const char* name;
				inst_size prev_nrof_instances;

				public:
				Step(const char* name, void* kernel, uint tpb, inst_size* nrof_instances, int* fp_lvl)
					: kernel(kernel), blockDim(dim3(tpb, 1, 1)), name(name) {{
					args[0] = (void*)nrof_instances;
					args[1] = (void*)fp_lvl;
					prev_nrof_instances = *nrof_instances;

					int numBlocksPerSm = 0;
					cudaDeviceProp deviceProp;
					cudaGetDeviceProperties(&deviceProp, 0);
					cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, blockDim.x, 0);
					max_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;

					updateDims();
				}}


				void updateDims(void) {{
					int wanted_blocks = (*((inst_size*)args[0]) + blockDim.x - 1) / blockDim.x;
					int used_blocks = min(max_blocks, wanted_blocks);

					gridDim.x = used_blocks;
					fprintf(stderr, \"Updated %s to %u/%u blocks of %u threads = %u threads.\\n\", name, used_blocks, max_blocks, blockDim.x, used_blocks * blockDim.x);
				}}

				void launch(void) {{
					if (*((inst_size*)args[0]) != prev_nrof_instances){{
						updateDims();
						prev_nrof_instances = *((inst_size*)args[0]);
					}}

					CHECK(
						cudaLaunchCooperativeKernel(
							kernel,
							gridDim,
							blockDim,
							args
						)
					);
				}}
			}};
		"};
		Some(cg + "\n" + &launch_params)
	}

	fn globals(&self) -> Option<String> {
		let mut result = self.program.inline_global_cpp.join("\n");
		result.push_str("\n");
		result.push_str(&self.fp.global_decl());
		result.push_str("size_t created_instances_offset = 0;");
		Some(result)
	}

	fn functions(&self) -> Option<String> {
		let mut functions = self.step_transpiler.functions();

		functions.push(self.copy_counters_from_struct_as_c());

		Some(functions.join("\n"))
	}

	fn kernels(&self) -> Option<String> {
		Some(
			self.program
				.structs
				.iter()
				.map(|strct| {
					strct
						.steps
						.iter()
						.map(|step| self.step_as_kernel(&strct.name, &step.name))
						.chain(std::iter::once(self.step_as_kernel(&strct.name, &"print".to_string())))
				})
				.flatten()
				.collect::<Vec<String>>()
				.join("\n")
		)
	}

	fn pre_main(&self) -> Option<String> {
		let mut struct_names: Vec<&String> = self.program.structs.iter().map(|s| &s.name).collect();
		struct_names.sort();
		let mut result = format!("\n\tcreated_instances_offset = host_{}.created_instances_offset();\n", struct_names.first().unwrap());

		// nrof_instances counters
		result.push_str(&struct_names.iter()
						 .enumerate()
						 .map(|(idx, strct)| formatdoc!("
							\tinst_size nrof_{strct}s = structs[{idx}].nrof_instances + 1;"
						 ))
						 .collect::<Vec<String>>()
						 .join("\n"));

		// fp_lvl passed to Step instances
		result.push_str("\n\tint fp_lvl = -1;\n");

		// The Step instances
		let print = "print".to_string();
		result.push_str(
			&self.program
				.structs
				.iter()
				.map(|strct|
					strct.steps
						 .iter()
						 .map(|step| &step.name)
						 .chain(std::iter::once(&print))
						 .map(|step_name|
							format!(
								"\tStep {strct_n}_{step_name}(\n\t\t\"{strct_n}.{step_name}\",\n\t\t(void*){kernel},\n\t\tTHREADS_PER_BLOCK,\n\t\t&nrof_{strct_n}s,\n\t\t&fp_lvl\n\t);",
								strct_n = strct.name,
								kernel = self.kernel_name(&strct.name, &step_name)
							)
						 )
				)
				.flatten()
				.collect::<Vec<String>>()
				.join("\n")
		);

		Some(result)
	}

	fn main(&self) -> Option<String> {
		Some(self.schedule_as_c(&self.program.schedule, 0))
	}

	fn post_main(&self) -> Option<String> {
		None
	}
}
