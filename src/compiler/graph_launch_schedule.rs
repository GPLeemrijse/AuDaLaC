use crate::analysis::constructors;
use std::collections::HashSet;
use crate::analysis::fixpoint_depth;
use crate::utils::format_signature;
use crate::analysis::get_step_to_structs;
use crate::compiler::compilation_traits::*;

use crate::compiler::StepBodyCompiler;
use crate::parser::ast::*;
use indoc::formatdoc;
use std::collections::BTreeSet;
use std::collections::HashMap;

pub struct GraphLaunchSchedule<'a> {
	program: &'a Program,
	fp: &'a dyn FPStrategy,
	step_to_structs: HashMap<&'a String, Vec<&'a ADLStruct>>,
	step_transpiler: &'a StepBodyCompiler<'a>
}

impl GraphLaunchSchedule<'_> {
	pub fn new<'a>(
		program: &'a Program,
		fp: &'a dyn FPStrategy,
		step_transpiler: &'a StepBodyCompiler<'_>,
	) -> GraphLaunchSchedule<'a> {
		GraphLaunchSchedule {
			program,
			fp,
			step_to_structs: get_step_to_structs(program),
			step_transpiler: step_transpiler
		}
	}

	fn kernel_name(&self, strct: &String, step: &String) -> String {
		format!("kernel_{strct}_{step}")
	}

	fn schedule_as_c(&self, sched : &Schedule, fp_lvl: usize) -> String {
		use Schedule::*;
		match sched {
			StepCall(..) => self.stepcall_as_c(sched, fp_lvl),
			TypedStepCall(..) => self.typedstepcall_as_c(sched, fp_lvl),
			Sequential(..) => self.sequential_as_c(sched, fp_lvl),
			Fixpoint(..) => self.fixpoint_as_c(sched, fp_lvl),
		}
	}

	fn stepcall_as_c(&self, stepcall : &Schedule, fp_lvl: usize) -> String {
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

			self.schedule_as_c(&combined_sched, fp_lvl)
		} else {
			unreachable!()
		}
	}


	fn typedstepcall_as_c(&self, typedstepcall : &Schedule, fp_lvl: usize) -> String {
		use Schedule::*;

		if let TypedStepCall(strct_name, step_name, _) = typedstepcall {
			let kernel = self.kernel_name(strct_name, step_name);
			let indent = "\t".repeat(fp_lvl + 1);
			let step = self.program.step_by_name(strct_name, step_name);
			let mut constructors : Vec<String> = if step.is_none() {
                Vec::new()
            } else {
                constructors(step.unwrap()).into_iter().cloned().collect()
            };
			let update_kernel = if !constructors.is_empty() {
				constructors.sort();
				let kernel = format!("update_nrof_{}", constructors.join("_"));
				format!("\n{indent}schedule.add_step((void*){kernel}, 1, 0);")
			} else {
				String::new()
			};

			let smem = 32 * 4;// 32 banks of 4 bytes

			formatdoc!{"
				{indent}schedule.add_step((void*){kernel}, {strct_name}_capacity, {smem});{update_kernel}"
			}
		} else {
			unreachable!();
		}
	}

	fn sequential_as_c(&self, sequential : &Schedule, fp_lvl: usize) -> String {
		use Schedule::*;
		if let Sequential(s1, s2, _) = sequential {
			let s1_as_c = self.schedule_as_c(s1, fp_lvl);
			let s2_as_c = self.schedule_as_c(s2, fp_lvl);
			formatdoc!("
				{s1_as_c}
				{s2_as_c}"
			)
		} else {
			unreachable!();
		}
	}

	fn fixpoint_as_c(&self, fixpoint : &Schedule, fp_lvl: usize) -> String {
		use Schedule::*;
		if let Fixpoint(s, _) = fixpoint {
			let sched = self.schedule_as_c(s, fp_lvl + 1);
			let indent = "\t".repeat(fp_lvl + 1);
			formatdoc! {"
				{indent}schedule.begin_fixpoint();
				{sched}
				{indent}schedule.end_fixpoint();
				"
			}

		} else {
			unreachable!();
		}
	}

	fn step_as_kernel(&self, strct: &String, step: &String) -> String {
		let kernel_name = self.kernel_name(strct, step);
		let func_header = format!("__global__ void {kernel_name}");

		let params = vec!["int fp_lvl".to_string()];

		let kernel_signature = format_signature(&func_header, &params, 0);

		formatdoc! {"
			{kernel_signature}
				const grid_group grid = this_grid();
				inst_size nrof_instances = {strct_n_lwr}->nrof_instances();
				{fp_pre}
				RefType self = grid.thread_rank();
				if (self < nrof_instances){{
					{f_name}(self, stable_ptr);
				}}
				{fp_post}
			}}
		",
			strct_n_lwr = strct.to_lowercase(),
			fp_pre = self.fp.top_of_kernel_decl(),
			f_name = self.step_transpiler.execute_f_name(strct, step),
			fp_post = self.fp.post_kernel(),
		}
	}

	fn relaunch_kernel(&self) -> String {
		let kernel_name = "relaunch_fp_kernel";
		let func_header = format!("__global__ void {kernel_name}");
		let params = vec!["int fp_lvl".to_string(), "cudaGraphExec_t restart".to_string(), "cudaGraphExec_t exit".to_string()];
		let kernel_signature = format_signature(&func_header, &params, 0);
		let is_stable = self.fp.is_stable(0);

		formatdoc! {"
			{kernel_signature}
			\tif(!{is_stable}){{
			\t\t{is_stable} = true;
			\t\tCHECK(cudaGraphLaunch(restart, cudaStreamGraphTailLaunch));
			\t}}
			\telse if (exit != NULL){{
			\t\tCHECK(cudaGraphLaunch(exit, cudaStreamGraphTailLaunch));
			\t}}
			}}
		"}
	}

	fn launch_kernel(&self) -> String {
		let kernel_name = "launch_kernel";
		let func_header = format!("__global__ void {kernel_name}");
		let params = vec!["cudaGraphExec_t graph".to_string()];
		let kernel_signature = format_signature(&func_header, &params, 0);

		formatdoc! {"
			{kernel_signature}
			\tCHECK(cudaGraphLaunch(graph, cudaStreamGraphTailLaunch));
			}}
		"}
	}

	fn update_counter_kernels(&self) -> String {
		let mut unique_updates = HashSet::new();
		self.unique_updates(&self.program.schedule, &mut unique_updates);
		unique_updates.iter().map(|u| {
			let kernel_name = format!("update_nrof_{}", u.join("_"));
			let func_header = format!("__global__ void {kernel_name}");
			let params = vec![];
			let kernel_signature = format_signature(&func_header, &params, 0);
			let updates = u.iter()
						   .map(|s| format!("\t{}->set_active_to_created();", s.to_lowercase()))
						   .collect::<Vec<String>>()
						   .join("\n");
			formatdoc!{"
				{kernel_signature}
				{updates}
				}}
			"}
		})
		.collect::<Vec<String>>()
		.join("\n")
	}

	fn unique_updates(&self, sched: &Schedule, updates: &mut HashSet<Vec<String>>) {
		use Schedule::*;
		match sched {
			StepCall(step_name, _) => {
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
				self.unique_updates(&combined_sched, updates);
			}
			TypedStepCall(struct_name, step_name, _) => {
				let step = self.program.step_by_name(struct_name, step_name);
				let mut constructors : Vec<String> = if step.is_none() {
                    Vec::new()
                } else {
                    constructors(step.unwrap()).into_iter().cloned().collect()
                };
				if !constructors.is_empty() {
					constructors.sort();
					updates.insert(constructors);
				}
			}
			Sequential(s1, s2, _) => {
				self.unique_updates(s1, updates);
				self.unique_updates(s2, updates);
			},
			Fixpoint(s, _) => self.unique_updates(s, updates),
		}
	}
}

impl CompileComponent for GraphLaunchSchedule<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<cooperative_groups.h>");
		set.insert("\"Schedule.h\"");
	}

	fn defines(&self) -> Option<String> {
		Some(format!("#define FP_DEPTH {}", fixpoint_depth(&self.program.schedule)))
	}

	fn typedefs(&self) -> Option<String> {
		let cg = "using namespace cooperative_groups;\n".to_string();
		Some(cg)
	}

	fn globals(&self) -> Option<String> {
		let mut result = self.program.inline_global_cpp.join("\n");
		result.push_str(&format!("\n{}", self.fp.global_decl()));
		Some(result)
	}

	fn functions(&self) -> Option<String> {
		let functions = self.step_transpiler.functions();

		Some(functions.join("\n"))
	}

	fn kernels(&self) -> Option<String> {
		let mut kernels : Vec<String> = self.program
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
										.collect();

		if fixpoint_depth(&self.program.schedule) > 0 {
			kernels.push(self.relaunch_kernel());
		}
		kernels.push(self.launch_kernel());
		kernels.push(self.update_counter_kernels());

		Some(
			kernels.join("\n")
		)
	}

	fn pre_main(&self) -> Option<String> {
		let relaunch_kernel = if fixpoint_depth(&self.program.schedule) > 0 {
			"(void*)relaunch_fp_kernel"
		} else {
			"NULL"
		};

		let mut result = formatdoc!{"
			\tcudaStream_t kernel_stream;
			\tCHECK(cudaStreamCreate(&kernel_stream));
			\tSchedule schedule((void*)launch_kernel, {relaunch_kernel});
		
		"};
		
		result.push_str(&self.schedule_as_c(&self.program.schedule, 0));


		result.push_str(&formatdoc!{"
			\tcudaGraphExec_t graph_exec = schedule.instantiate(kernel_stream);
			//\tschedule.print_dot();"
		});

		Some(result)
	}

	fn main(&self) -> Option<String> {
		Some(formatdoc!{"
			\tCHECK(cudaGraphLaunch(graph_exec, kernel_stream));"
		})
	}

	fn post_main(&self) -> Option<String> {
		None
	}
}
