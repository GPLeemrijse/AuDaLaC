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
	step_to_structs: HashMap<&'a String, Vec<&'a ADLStruct>>,
	step_transpiler: &'a StepBodyCompiler<'a>,
}

impl GraphLaunchSchedule<'_> {
	pub fn new<'a>(
		program: &'a Program,
		step_transpiler: &'a StepBodyCompiler<'_>,
	) -> GraphLaunchSchedule<'a> {
		GraphLaunchSchedule {
			program,
			step_to_structs: get_step_to_structs(program),
			step_transpiler: step_transpiler,
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
			formatdoc!{"
				{indent}schedule.add_step((void*){kernel}, {strct_name}_capacity);"
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

		let execute_step = formatdoc!{"
			\tinst_size nrof_instances = {strct_n_lwr}->nrof_instances();
			\tRefType self = grid.thread_rank();
			\tif (self >= nrof_instances)
			\t\treturn;
			\t{f_name}(self, &stable);",
			f_name = self.step_transpiler.execute_f_name(strct, step),
			strct_n_lwr = strct.to_lowercase()
		};

		formatdoc! {"
			{kernel_signature}
				const grid_group grid = this_grid();
				bool stable = true;
			{execute_step}
				if(!stable && fp_lvl >= 0){{
					clear_stack(fp_lvl);
				}}
			}}
		"}
	}

	fn relaunch_kernel(&self) -> String {
		let kernel_name = "relaunch_fp_kernel";
		let func_header = format!("__global__ void {kernel_name}");
		let params = vec!["int fp_lvl".to_string(), "cudaGraphExec_t restart".to_string(), "cudaGraphExec_t exit".to_string()];
		let kernel_signature = format_signature(&func_header, &params, 0);

		formatdoc! {"
			{kernel_signature}
			\tif(!fp_stack[fp_lvl]){{
			\t\tfp_stack[fp_lvl] = true;
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
		result.push_str(&formatdoc!{"
			
			__device__ volatile bool fp_stack[FP_DEPTH];

			__device__ void clear_stack(int lvl){{
				while(lvl >= 0){{
					fp_stack[lvl--] = false;
				}}
			}}
		"});
		Some(result)
	}

	fn functions(&self) -> Option<String> {
		let functions = self.step_transpiler.functions();

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
				.chain(std::iter::once(self.relaunch_kernel()))
				.chain(std::iter::once(self.launch_kernel()))
				.collect::<Vec<String>>()
				.join("\n")
		)
	}

	fn pre_main(&self) -> Option<String> {
		let mut result = formatdoc!{"
			\tcudaStream_t kernel_stream;
			\tCHECK(cudaStreamCreate(&kernel_stream));
			\tSchedule schedule((void*)launch_kernel, (void*)relaunch_fp_kernel);
		
		"};
		
		result.push_str(&self.schedule_as_c(&self.program.schedule, 0));


		result.push_str(&formatdoc!{"
			\tcudaGraphExec_t graph_exec = schedule.instantiate(kernel_stream);
			\tschedule.print();"
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
