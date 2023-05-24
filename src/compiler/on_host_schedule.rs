use crate::utils::as_printf;
use crate::utils::format_signature;
use crate::in_kernel_compiler::StepBodyCompiler;
use std::collections::HashMap;
use crate::parser::ast::*;
use std::collections::BTreeSet;
use crate::compiler::compilation_traits::*;
use indoc::formatdoc;

pub struct OnHostSchedule<'a> {
	program : &'a Program,
	fp : &'a dyn FPStrategy,
	step_to_structs : HashMap<&'a String, Vec<&'a ADLStruct>>,
	instances_per_thread : usize,
	threads_per_block : usize,
	nrof_threads : usize,
	step_transpiler : &'a StepBodyCompiler<'a>
}

impl OnHostSchedule<'_> {
	
	pub fn new<'a>(program : &'a Program,
				   fp : &'a dyn FPStrategy,
				   instances_per_thread: usize,
				   threads_per_block : usize,
				   nrof_threads : usize,
				   step_transpiler : &'a StepBodyCompiler<'_>) -> OnHostSchedule<'a> {
		OnHostSchedule {
			program,
			fp,
			step_to_structs: program.get_step_to_structs(),
			instances_per_thread,
			threads_per_block,
			nrof_threads,
			step_transpiler : step_transpiler,
		}
	}

	fn step_call_function_as_c(&self, strct: &ADLStruct, step: &Step) -> String {
		let strct_name = &strct.name;
		let strct_name_lwr = strct.name.to_lowercase();
		let step_name = &step.name;
		let kernel_name = format!("{strct_name}_{step_name}");

		formatdoc!("
			void launch_{kernel_name}(bool update_nrof_instances) {{
				if (update_nrof_instances){{
					{strct_name}* manager;
					CHECK(
						cudaGetSymbolAddress((void**)&manager, {strct_name_lwr})
					);
					inst_size* active_instances = &(manager->active_instances);
					inst_size* instantiated_instances = &(manager->instantiated_instances);

					CHECK(
						cudaMemcpy(&active_{strct_name}_instances, instantiated_instances, sizeof(inst_size), cudaMemcpyDeviceToHost)
					);

					CHECK(
						cudaMemcpy(active_instances, instantiated_instances, sizeof(inst_size), cudaMemcpyDeviceToDevice)
					);
				}}

				inst_size nrof_instances = active_{strct_name}_instances;
				void* args[] = {{}};
				auto dims = ADL::get_launch_dims(nrof_instances, (void*){kernel_name});

				CHECK(
					cudaLaunchCooperativeKernel(
						(void*){kernel_name},
						std::get<0>(dims),
						std::get<1>(dims),
						args
					)
				);
			}}
		")
	}
}

impl CompileComponent for OnHostSchedule<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<stdio.h>");
		set.insert("<cooperative_groups.h>");
	}

	fn defines(&self) -> Option<String> {
		let tpb = self.threads_per_block;
		Some(formatdoc!{"
			#define FP_DEPTH {}
			#define THREADS_PER_BLOCK {tpb}
		", self.program.schedule.fixpoint_depth()})
	}
	
	fn typedefs(&self) -> Option<String> {
		Some("using namespace cooperative_groups;".to_string())
	}

	fn globals(&self) -> Option<String> {
		let mut result = self.fp.global_decl();

		let active_instances = self.program
								   .structs
								   .iter()
								   .map(|s|
										format!("inst_size active_{}_instances;", s.name)
								   )
								   .collect::<Vec<String>>()
								   .join("\n");
		
		result.push_str(active_instances);
		Some(result)
	}

	fn functions(&self) -> Option<String> {
		self.program.structs.iter()
						 	.map(|strct|
						 		strct.steps.iter()
										   .map(|step| self.step_call_function_as_c(strct, step))
						 	)
						 	.flatten()
						 	.collect::<Vec<String>>()
						 	.join("\n")
	}

	fn kernels(&self) -> Option<String> {
		None
	}
	
	fn pre_main(&self) -> Option<String> {
		None
	}
	
	fn main(&self) -> Option<String> {
		None
	}

	fn post_main(&self) -> Option<String> {
		None
	}
}