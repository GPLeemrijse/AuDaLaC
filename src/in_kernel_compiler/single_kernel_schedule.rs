use crate::utils::as_printf;
use crate::utils::format_signature;
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
	threads_per_block : usize,
	nrof_threads : usize,
	step_transpiler : &'a StepBodyTranspiler<'a>
}

impl SingleKernelSchedule<'_> {
	const KERNEL_NAME : &str = "schedule_kernel";

	pub fn new<'a>(program : &'a Program,
				   fp : &'a dyn FPStrategy,
				   instances_per_thread: usize,
				   threads_per_block : usize,
				   nrof_threads : usize,
				   step_transpiler : &'a StepBodyTranspiler<'_>) -> SingleKernelSchedule<'a> {
		SingleKernelSchedule {
			program,
			fp,
			step_to_structs: program.get_step_to_structs(),
			instances_per_thread,
			threads_per_block,
			nrof_threads,
			step_transpiler : step_transpiler,
		}
	}

	fn _kernel_arguments(&self) -> Vec<String> {
		self.program.structs.iter()
							.map(|s| format!("&gm_{}", s.name))
							.collect()
	}

	fn _get_struct_manager_parameters(&self) -> Vec<String> {
		self.program.structs.iter()
							.map(|s| format!("{} * const __restrict__ {}", s.name, s.name.to_lowercase()))
							.collect()
	}

	fn kernel_parameters(&self, including_self : bool) -> Vec<String> {
		// Use this version when passing struct managers as kernel parameters
		// let mut structs = self._get_struct_manager_parameters();
		let mut structs = Vec::new();

		if including_self {
			let mut self_and_structs = vec!["const RefType self".to_string()];
			self_and_structs.append(&mut structs);
			self_and_structs
		} else {
			structs
		}
	}

	fn step_call_arguments(&self) -> Vec<String> {
		/*
		Use this when not using global struct managers
		self.program.structs.iter()
							.map(|s| s.name.to_lowercase())
							.collect::<Vec<String>>()
		*/
		Vec::new()
	}

	fn kernel_schedule_body(&self) -> String {
		let ind = "\t";
		let stab_stack = self.fp.top_of_kernel_decl();

		let schedule = self.schedule_as_c(&self.program.schedule, 0);

		formatdoc!{"
			{ind}const grid_group grid = this_grid();
			{ind}const thread_block block = this_thread_block();
			{ind}const uint in_grid_rank = grid.thread_rank();
			{ind}const uint in_block_rank = block.thread_rank();
			{ind}const uint block_idx = grid.block_rank();
			{ind}const uint block_size = block.size();
			{ind}inst_size nrof_instances;
			{ind}bool step_parity = false;
			{ind}RefType self;
			{stab_stack}

			{schedule}"
		}
	}

	fn schedule_as_c(&self, sched : &Schedule, fp_level : usize) -> String {
		use crate::ast::Schedule::*;
		
		match sched {
			StepCall(..) => self.step_call_as_c(sched, fp_level),
			Sequential(..) => self.sequential_step_as_c(sched, fp_level),
			TypedStepCall(..) => self.typed_step_call_as_c(sched, fp_level),
			Fixpoint(..) => self.fixpoint_as_c(sched, fp_level),
		}
	}

	fn fixpoint_as_c(&self, sched : &Schedule, fp_level : usize) -> String {
		use crate::ast::Schedule::*;
		if let Fixpoint(s, _) = sched {
			let indent = "\t".repeat(fp_level+1);
			let pre_iteration = self.fp.pre_iteration(fp_level);
			let sched = self.schedule_as_c(s, fp_level + 1);
			let post_iteration = self.fp.post_iteration(fp_level);
			let is_stable = self.fp.is_stable(fp_level);

			formatdoc!{"
				{indent}do{{
				{pre_iteration}
				{sched}
				{post_iteration}
				{indent}	grid.sync();
				{indent}}} while(!{is_stable});
			"}
		} else {
			unreachable!()
		}
	}

	fn sequential_step_as_c(&self, sched : &Schedule, fp_level : usize) -> String {
		use crate::ast::Schedule::*;
		if let Sequential(s1, s2, _) = sched {
			let indent = "\t".repeat(fp_level+1);
			let sched1 = self.schedule_as_c(s1, fp_level);
			let sched2 = self.schedule_as_c(s2, fp_level);

			formatdoc!{"
					{sched1}

					{indent}grid.sync();

					{sched2}"
			}
		} else {
			unreachable!()
		}
	}

	fn typed_step_call_as_c(&self, sched : &Schedule, fp_level : usize) -> String {
		use crate::ast::Schedule::*;

		if let TypedStepCall(struct_name, step_name, _) = sched {
			let indent = "\t".repeat(fp_level+1);
			let strct = self.program.struct_by_name(struct_name).unwrap();
			let step_call;

			if step_name == "print" {
				step_call = self.call_print_function(strct);
			} else {
				let step = strct.step_by_name(step_name).unwrap();
				step_call = self.call_step_function(strct, step, fp_level);
			}

			let set_unstable = self.fp.set_unstable(fp_level);
			let nrof_instances = format!("{}->nrof_instances2(step_parity)", struct_name.to_lowercase());

			let step_call_and_stable_set = if fp_level == 0 {
				format!("{indent}\t{step_call};")
			} else {
				formatdoc!{"
					{indent}\tif (!{step_call}) {{
					{indent}\t\t{set_unstable}	
					{indent}\t}}"
				}
			};

			formatdoc!{"
				{indent}nrof_instances = {nrof_instances};
				{indent}for(int i = 0; i < INSTS_PER_THREAD; i++){{
				{indent}	self = block_size * (i + block_idx * INSTS_PER_THREAD) + in_block_rank;
				{indent}	if (self >= nrof_instances) break;
				
				{step_call_and_stable_set}				
				{indent}}}
				{indent}step_parity = !step_parity;"
			}
		} else {
			unreachable!()
		}
	}

	fn step_call_as_c(&self, sched : & Schedule, fp_level : usize) -> String {
		use crate::ast::Schedule::*;
		if let StepCall(step_name, _) = sched {
			let structs = self.step_to_structs
							  .get(step_name)
							  .unwrap();
			if structs.len() > 1 {
				unimplemented!("Untyped stepcalls with more than 1 implementor are not implemented yet.");
			}

			// For now fall back on single-struct steps
			let strct = structs[0];
			self.typed_step_call_as_c(&TypedStepCall(
				strct.name.clone(),
				step_name.to_string(),
				(0, 0)
			), fp_level)
		} else {
			unreachable!()
		}
	}

	fn step_as_c_function(&self, strct : &ADLStruct, step : &Step, fp_level : usize) -> String {
		let func_name = self.step_function_name(strct, step);
		let func_header = format!("__device__ __inline__ bool {func_name}");

		let mut params = self.kernel_parameters(true);
		params.push("bool step_parity".to_string());

		let kernel_signature = format_signature(&func_header, params, 0);

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

	fn print_as_c_function(&self, strct : &ADLStruct, fp_level : usize) -> String {
		let func_name = format!("print_{}", strct.name);
		let func_header = format!("__device__ __inline__ bool {func_name}");

		let params = self.kernel_parameters(true);

		let kernel_signature = format_signature(&func_header, params, 0);

		let pre_step_fp = self.fp.pre_step_function(fp_level);
		let post_step_fp = self.fp.post_step_function(fp_level);

		let s_name = &strct.name;
		let s_name_lwr = s_name.to_lowercase();
		let params_as_fmt = strct.parameters.iter()
											.map(|(n, t, _)| format!("{n}={}", as_printf(t)))
											.reduce(|a, n| a + ", " + &n)
											.unwrap();

		let params_as_expr = strct.parameters.iter()
											.map(|(n, _, _)| format!(", LOAD({s_name_lwr}->{n}[self])"))
											.reduce(|a, n| a + &n)
											.unwrap();

		formatdoc!{"
			{kernel_signature}
				{pre_step_fp}
				if (self != 0) {{
					printf(\"{s_name}(%u): {params_as_fmt}\\n\", self{params_as_expr});
				}}
				{post_step_fp}
			}}
		"}
	}


	fn call_step_function(&self, strct : &ADLStruct, step : &Step, _fp_level : usize) -> String {
		let func_name = self.step_function_name(strct, step);
		let mut func_args = vec!["self".to_string()];

		func_args.append(&mut self.step_call_arguments());
		func_args.push("step_parity".to_string());

		let args = func_args.join(", ");
		formatdoc!{"
			{func_name}({args})"}
	}

	fn call_print_function(&self, strct : &ADLStruct) -> String {
		let func_name = format!("print_{}", strct.name);
		let mut func_args = vec!["self".to_string()];

		func_args.append(&mut self.step_call_arguments());

		let args = func_args.join(", ");
		formatdoc!{"
			{func_name}({args})"}
	}

	fn step_function_name(&self, strct : &ADLStruct, step : &Step) -> String {
		format!("{}_{}", strct.name, step.name)
	}
}

impl CompileComponent for SingleKernelSchedule<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<cooperative_groups.h>");
	}

	fn defines(&self) -> Option<String> {
		Some(formatdoc!("
				#define THREADS_PER_BLOCK {}
				#define INSTS_PER_THREAD {}
				#define FP_DEPTH {}
			",
			self.threads_per_block,
			self.instances_per_thread,
			self.program.schedule.fixpoint_depth()
		))
	}
	
	fn typedefs(&self) -> Option<String> {
		Some("using namespace cooperative_groups;".to_string())
	}

	fn globals(&self) -> Option<String> {
		Some(self.fp.global_decl())
	}

	fn functions(&self) -> Option<String> {
		let mut functions = self.step_transpiler.functions();

		let mut step_functions = self.program.structs
								.iter()
								.map(|strct|
									std::iter::once(self.print_as_c_function(strct, 666))
									.chain(
										strct.steps
											 .iter()
											 .map(|step|
											 	self.step_as_c_function(strct, step, 666)
											 )
									)
								)
								.flatten()
								.collect::<Vec<String>>();

		functions.append(&mut step_functions);

		Some(functions.join("\n"))
	}

	fn kernels(&self) -> Option<String> {
		let kernel_header = format!("__global__ void {}", SingleKernelSchedule::KERNEL_NAME);
		let kernel_signature = format_signature(&kernel_header, self.kernel_parameters(false), 0);
		
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
		let nrof_threads =  self.nrof_threads;

		Some(formatdoc!{"
			\tvoid* {kernel_name}_args[] = {{}};
			\tauto dims = ADL::get_launch_dims({nrof_threads}, (void*){kernel_name});

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