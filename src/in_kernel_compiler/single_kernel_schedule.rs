use crate::in_kernel_compiler::StepBodyTranspiler;
use std::collections::HashMap;
use crate::ast::*;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;

pub struct SingleKernelSchedule<'a> {
	program : &'a Program,
	fp : &'a dyn FPStrategy,
	step_to_structs : HashMap<&'a String, Vec<&'a ADLStruct>>,
	instances_per_thread : usize,
	step_transpiler : &'a StepBodyTranspiler<'a>
}

impl SingleKernelSchedule<'_> {
	const KERNEL_NAME : &str = "schedule_kernel";

	pub fn new<'a>(program : &'a Program, fp : &'a dyn FPStrategy, instances_per_thread: usize, step_transpiler : &'a StepBodyTranspiler<'_>) -> SingleKernelSchedule<'a> {
		SingleKernelSchedule {
			program,
			fp,
			step_to_structs: program.get_step_to_structs(),
			instances_per_thread,
			step_transpiler : step_transpiler,
		}
	}

	fn kernel_arguments(&self) -> Vec<String> {
		self.program.structs.iter()
							.map(|s| format!("&gm_{}", s.name))
		 					.collect()
	}

	fn kernel_parameters(&self) -> Vec<String> {
		self.program.structs.iter()
							.map(|s| format!("{}* {}", s.name, s.name.to_lowercase()))
		 					.collect()
	}

	fn step_call_arguments(&self) -> String {
		self.program.structs.iter()
							.map(|s| s.name.to_lowercase())
							.collect::<Vec<String>>()
							.join(", ")
	}

	fn kernel_schedule_body(&self) -> String {
		let ind = "\t";
		let stab_stack = self.fp.top_of_kernel_decl();

		let schedule = self.schedule_as_c(&self.program.schedule, 0);

		formatdoc!{"
			{ind}{stab_stack}
			{ind}grid_group grid = this_grid();
			{ind}thread_block block = this_thread_block();
			{ind}unsigned int block_rank = block.thread_rank();
			{ind}unsigned int block_idx = grid.block_rank();
			{ind}unsigned int block_size = block.size();
			{ind}const inst_size nrof_instances;

			{schedule}"
		}
	}

	fn schedule_as_c(&self, sched : &Schedule, fp_level : usize) -> String {
		use crate::ast::Schedule::*;
		let indent = "\t".repeat(fp_level+1);
		
		match sched {
			StepCall(step_name, _) => {
				let structs = self.step_to_structs
								  .get(step_name)
								  .unwrap();
				if structs.len() > 1 {
					unimplemented!("Untyped stepcalls with more than 1 implementor are not allowed.");
				}

				// For now fall back on single-struct steps
				let strct = structs[0];
				self.schedule_as_c(&TypedStepCall(
					strct.name.clone(),
					step_name.to_string(),
					(0, 0)
				), fp_level)
			},
			Sequential(s1, s2, _) => {
				let sched1 = self.schedule_as_c(s1, fp_level);
				let sched2 = self.schedule_as_c(s2, fp_level);

				formatdoc!{"
					{sched1}

					{indent}grid.sync();

					{sched2}"}
			},
			TypedStepCall(struct_name, step_name, _) => {
				let strct = self.program.struct_by_name(struct_name).unwrap();
				let step_call = if step_name == "print" {
					format!("Body: {}.print", struct_name)
				} else {
					let step = strct.step_by_name(step_name).unwrap();
					self.call_step_function(strct, step, fp_level)
				};

				let nrof_instances = format!("{}->nrof_instances()", struct_name.to_lowercase());
				let set_unstable = self.fp.set_unstable(fp_level);
				formatdoc!{"
					{indent}nrof_instances = {nrof_instances};
					{indent}#pragma unroll
					{indent}for(int i = 0; i < INSTS_PER_THREAD; i++){{
					{indent}	RefType self = block_size * (i + block_idx * INSTS_PER_THREAD) + block_rank;
					{indent}	if (self >= nrof_instances) break;

					{indent}	if (!{step_call}) {{
					{indent}		{set_unstable}	
					{indent}	}}
					{indent}}}"
				}
			},
			Fixpoint(s, _) => {
				let pre_iteration = self.fp.pre_iteration(fp_level);
				let sched = self.schedule_as_c(s, fp_level + 1);
				let post_iteration = self.fp.post_iteration(fp_level);
				let is_stable = self.fp.is_stable(fp_level);

				return formatdoc!{"
					{indent}do{{
					{pre_iteration}
					{sched}
					{post_iteration}
					{indent}	grid.sync();
					{indent}}} while(!{is_stable});
				"};
			}
		}
	}

	fn step_as_c_function(&self, strct : &ADLStruct, step : &Step, fp_level : usize) -> String {
		let func_name = self.step_function_name(strct, step);
		let func_header = format!("__device__ bool {func_name}");
		let kernel_signature = self.format_signature(&func_header, self.kernel_parameters());

		let step_body = self.step_transpiler.statements_as_c(&step.statements, strct, step, 1, fp_level);
		let pre_step_fp = self.fp.pre_step_function(fp_level);
		let post_step_fp = self.fp.post_step_function(fp_level);


		formatdoc!{"
			{kernel_signature}
				{pre_step_fp}
				{step_body}
				{post_step_fp}
			}}
		"}
	}

	fn call_step_function(&self, strct : &ADLStruct, step : &Step, _fp_level : usize) -> String {
		let func_name = self.step_function_name(strct, step);
		let func_args = self.step_call_arguments();
		formatdoc!{"
			{func_name}({func_args})"}
	}

	fn format_signature(&self, sig : &String, params : Vec<String>) -> String
	{
		let indent = format!(",\n{}{}",
						     "\t".repeat((sig.len()+1) / 4),
						     " ".repeat((sig.len()+1) % 4)
						    );
		format!("{sig}({}){{", params.join(&indent))
	}

	fn step_function_name(&self, strct : &ADLStruct, step : &Step) -> String {
		format!("{}_{}", strct.name, step.name)
	}
}

impl CompileComponent for SingleKernelSchedule<'_> {
	fn add_includes(&self, _set: &mut BTreeSet<&str>) {
	}

	fn defines(&self) -> Option<String> {
		Some(formatdoc!("
			#define INSTS_PER_THREAD {}
		", self.instances_per_thread))
	}
	
	fn typedefs(&self) -> Option<String> {
		None
	}
	
	fn globals(&self) -> Option<String> {
		Some(self.fp.global_decl())
	}
	
	fn functions(&self) -> Option<String> {
		Some(self.program.structs
					.iter()
					.map(|strct|
						strct.steps
							 .iter()
							 .map(|step|
							 	self.step_as_c_function(strct, step, 666)
							 )
					)
					.flatten()
					.collect::<Vec<String>>()
					.join("\n")
		)
	}

	fn kernels(&self) -> Option<String> {
		let kernel_header = format!("__global__ void {}", SingleKernelSchedule::KERNEL_NAME);
		let kernel_signature = self.format_signature(&kernel_header, self.kernel_parameters());
		
		let kernel_schedule_body = self.kernel_schedule_body();

		Some(formatdoc!{"
			{kernel_signature}
			{kernel_schedule_body}
			}}
		"})
	}
	
	fn pre_main(&self) -> Option<String> {
		None
	}
	
	fn main(&self) -> Option<String> {
		let kernel_name = SingleKernelSchedule::KERNEL_NAME;
		let arg_array = self.kernel_arguments().join(&format!(",\n\t\t"));

		Some(formatdoc!{"
			\tinst_size nrof_instances = 10000; // TODO
			\tvoid* {kernel_name}_args[] = {{
			\t	{arg_array}
			\t}};
			\tauto dims = ADL::get_launch_dims(nrof_instances, (void*){kernel_name});

			\tCHECK(
			\t	cudaLaunchCooperativeKernel(
			\t		(void*){kernel_name},
			\t		std::get<0>(dims),
			\t		std::get<1>(dims),
			\t		{kernel_name}_args
			\t	)
			\t);
			\tCHECK(cudaDeviceSynchronize());
		"})
	}
	
	fn post_main(&self) -> Option<String> {
		None
	}
}