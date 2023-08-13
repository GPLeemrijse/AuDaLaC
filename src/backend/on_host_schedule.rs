use crate::analysis::constructors;
use crate::utils::format_signature;
use crate::analysis::get_step_to_structs;
use crate::backend::compilation_traits::*;
use crate::backend::StepBodyCompiler;
use crate::frontend::ast::*;
use indoc::formatdoc;
use std::collections::BTreeSet;
use std::collections::HashMap;

pub struct OnHostSchedule<'a> {
	program: &'a Program,
	fp: &'a dyn FPStrategy,
	step_to_structs: HashMap<&'a String, Vec<&'a ADLStruct>>,
	step_transpiler: &'a StepBodyCompiler<'a>
}

impl OnHostSchedule<'_> {
	pub fn new<'a>(
		program: &'a Program,
		fp: &'a dyn FPStrategy,
		step_transpiler: &'a StepBodyCompiler<'_>
	) -> OnHostSchedule<'a> {
		OnHostSchedule {
			program,
			fp,
			step_to_structs: get_step_to_structs(program),
			step_transpiler: step_transpiler
		}
	}

	fn kernel_name(&self, strct: &String, step: &String) -> String {
		format!("kernel_{strct}_{step}")
	}

	fn copy_counters_from_struct_as_c(&self) -> String {
		formatdoc!{"
			__host__ inst_size get_created_instances_from(void* struct_ptr, cudaStream_t stream) {{
				cuda::atomic<inst_size, cuda::thread_scope_device> dest;
				CHECK(
					cudaMemcpyAsync(
						&dest,
						(void*)((size_t)struct_ptr + created_instances_offset),
						sizeof(cuda::atomic<inst_size, cuda::thread_scope_device>),
						cudaMemcpyDeviceToHost,
						stream
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
									"\n{indent}nrof_{strct}s = get_created_instances_from(loc_{}, kernel_stream);",
									strct.to_lowercase()
								)
							)
							.collect::<Vec<String>>()
							.join("\n")
			} else {
				String::new()
			};

			formatdoc!{"
				{indent}{strct_name}_{step_name}.launch(kernel_stream);{post_launch}
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
		let func_header = format!("__global__ void {kernel_name}");
		
		let mut params = vec!["inst_size nrof_instances".to_string(), "int fp_lvl".to_string()];
		
		let rotating = self.fp.is_stable(0).contains("iter_idx"); // Hacky...
		if rotating {
			params.push("iter_idx_t iter_idx".to_string());
		}
		let kernel_signature = format_signature(&func_header, &params, 0);

		let pre = self.fp.top_of_kernel_decl();
		let post = self.fp.post_kernel();

		formatdoc! {"
			{kernel_signature}
				const grid_group grid = this_grid();
				const thread_block block = this_thread_block();{pre}
				RefType self = grid.thread_rank();
				if (self < nrof_instances) {{
					{f_name}(self, &stable);
				}}{post}
			}}",
			f_name = self.step_transpiler.execute_f_name(strct, step)
		}
	}
}

impl CompileComponent for OnHostSchedule<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<cooperative_groups.h>");
		set.insert("\"OnHostStep.h\"");
	}

	fn defines(&self) -> Option<String> {
		None
	}

	fn typedefs(&self) -> Option<String> {
		let cg = "using namespace cooperative_groups;".to_string();
		let fp = self.fp.global_decl();
		
		Some(cg + "\n" + &fp)
	}

	fn globals(&self) -> Option<String> {
		let mut result = self.program.inline_global_cpp.join("\n");
		result.push_str("\n");
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
		let mut result = formatdoc!{"
			\tcreated_instances_offset = host_{}.created_instances_offset();
			\tcudaStream_t kernel_stream;
			\tcudaStreamCreate(&kernel_stream);
			", struct_names.first().unwrap()
		};

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
		result.push_str(&self.fp.initialise());

		// The Step instances
		let rotating = self.fp.is_stable(0).contains("iter_idx"); // Hacky...
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
						 	
							formatdoc!{"
								\tOnHostStep {strct_n}_{step_name}(
								\t\t\"{strct_n}.{step_name}\",
								\t\t(void*){kernel},
								\t\t&nrof_{strct_n}s,
								\t\t&fp_lvl,
								\t\t{iter_idx_arg}
								\t);",
								strct_n = strct.name,
								kernel = self.kernel_name(&strct.name, &step_name),
								iter_idx_arg = if rotating {"(void*)&iter_idx"} else {"NULL"},
							}
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
