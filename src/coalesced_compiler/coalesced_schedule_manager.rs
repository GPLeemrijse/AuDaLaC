use std::collections::HashMap;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;
use indoc::{formatdoc};


pub struct CoalescedScheduleManager<'a> {
	program : &'a Program,
	step_to_structs : HashMap<&'a String, Vec<&'a ADLStruct>>,
	struct_manager: &'a dyn StructManager,
	tpb: usize
}

impl ScheduleManager for CoalescedScheduleManager<'_> {
	fn add_includes(&self, set: &mut BTreeSet<std::string::String>) {
		set.insert("<stdio.h>".to_string());
		set.insert("\"fp_manager.h\"".to_string());
		set.insert("<cooperative_groups.h>".to_string());
	}
	fn defines(&self) -> std::string::String {
		format!("#define FP_DEPTH {}\n", self.program.schedule.fixpoint_depth())
	}
	fn struct_typedef(&self) -> std::string::String {
		String::new()
	}
	fn globals(&self) -> std::string::String {
		String::new()
	}
	fn function_defs(&self) -> std::string::String {
		String::new()
	}
	fn run_schedule(&self) -> std::string::String {
		let schedule = self.schedule_as_c(&self.program.schedule, 1);
		formatdoc!{"
			FPManager host_FP = FPManager(FP_DEPTH); // initially not done
				FPManager* device_FP = host_FP.to_device();

			{schedule}
		"}
	}
}


impl CoalescedScheduleManager<'_> {
	pub fn new<'a>(program: &'a Program, struct_manager : &'a dyn StructManager) -> CoalescedScheduleManager<'a> {
		CoalescedScheduleManager {
			program,
			step_to_structs : CoalescedScheduleManager::get_step_to_structs(program),
			struct_manager,
			tpb: 512
		}
	}

	fn step_call_as_c(&self, strct: &ADLStruct, step: &Step, indent_lvl : usize) -> String {
		let indent = "\t".repeat(indent_lvl);
		let strct_name = &strct.name;
		let tpb = self.tpb;

		let arg_array = self.struct_manager.kernel_arguments(strct, step)
										   .join(&format!(",\n{indent}\t"));
		let kernel_name = self.struct_manager.kernel_name(strct, step);

		formatdoc!("
			{indent}void* {kernel_name}_args[] = {{
			{indent}	{arg_array}
			{indent}}};
			{indent}CHECK(
			{indent}	cudaLaunchCooperativeKernel(
			{indent}		(void*){kernel_name},
			{indent}		dim3((host_{strct_name}.nrof_instances() + {tpb} - 1)/{tpb}),
			{indent}		dim3({tpb}),
			{indent}		{kernel_name}_args
			{indent}	)
			{indent});
			{indent}CHECK(cudaDeviceSynchronize());
		")
	}

	fn call_print_as_c(&self, struct_name: &String, indent_lvl : usize) -> String {
		let indent = "\t".repeat(indent_lvl);
		let tpb = self.tpb;
		let kernel_name = format!("{struct_name}_print");

		formatdoc!("
			{indent}{kernel_name}<<<(host_{struct_name}.nrof_instances() + {tpb} - 1)/{tpb}, {tpb}>>>(gm_{struct_name});
			{indent}CHECK(cudaDeviceSynchronize());
		")
	}

	fn schedule_as_c(&self, sched : &Schedule, indent_lvl : usize) -> String {
		use crate::ast::Schedule::*;
		let indent = "\t".repeat(indent_lvl);
		
		match sched {
			StepCall(step_name, _) => {
				self.step_to_structs.get(step_name)
									.unwrap()
									.iter() // iter of structs
									.map(|s| self.step_call_as_c(
												s,
												s.step_by_name(step_name).unwrap(),
												indent_lvl
											 )
									)
									.fold("".to_string(), |acc: String, nxt| acc + &nxt)
			},
			Sequential(s1, s2, _) => {
				format!("{}\n{}", self.schedule_as_c(s1, indent_lvl), self.schedule_as_c(s2, indent_lvl))
			},
			TypedStepCall(struct_name, step_name, _) => {
				let strct = self.program.struct_by_name(struct_name).unwrap();
				match strct.step_by_name(step_name) {
					Some(step) => self.step_call_as_c(
						strct,
						step,
						indent_lvl
					),
					None => {
						debug_assert!(step_name == "print");
						self.call_print_as_c(struct_name, indent_lvl)
					}
				}
			},
			Fixpoint(s, _) => {
				let sched = self.schedule_as_c(s, indent_lvl + 1);

				return formatdoc!{"
					{indent}host_FP.push();
					{indent}do{{
					{indent}	host_FP.reset();
					{indent}	host_FP.copy_to(device_FP);
					{sched}
					{indent}	host_FP.copy_from(device_FP);
					{indent}	if(!host_FP.done()) host_FP.clear();
					{indent}}}
					{indent}while(!host_FP.done());
					{indent}host_FP.pop();
				"};
			}
		}
	}

	fn get_step_to_structs(program: &Program) -> HashMap<&String, Vec<&ADLStruct>> {
		let mut s2s : HashMap<&String, Vec<&ADLStruct>> = HashMap::new();

		for strct in &program.structs {
			for step in &strct.steps {
				s2s.entry(&step.name).and_modify(|v| v.push(strct)).or_insert(vec![strct]);
			}
		}
		return s2s
	}
}