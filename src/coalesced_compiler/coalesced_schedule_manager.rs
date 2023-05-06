use std::collections::HashMap;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;
use indoc::{formatdoc};


pub struct CoalescedScheduleManager<'a> {
	program : &'a Program,
	step_to_structs : HashMap<&'a String, Vec<&'a ADLStruct>>,
	struct_manager: &'a dyn StructManager,
	tpb: usize,
	printnthinst: usize,
	print_unstable: bool
}

impl ScheduleManager for CoalescedScheduleManager<'_> {
	fn add_includes(&self, set: &mut BTreeSet<&str>) {
		set.insert("<stdio.h>");
		set.insert("\"fp_manager.h\"");
		set.insert("<cooperative_groups.h>");
	}

	fn defines(&self) -> String {
		let tpb = self.tpb;
		formatdoc!{"
			#define FP_DEPTH {}
			#define THREADS_PER_BLOCK {tpb}
		", self.program.schedule.fixpoint_depth()}
	}

	fn struct_typedef(&self) -> String {
		String::new()
	}

	fn globals(&self) -> String {
		formatdoc!{"
			FPManager host_FP = FPManager(FP_DEPTH);
			FPManager* device_FP;

		"}
	}

	fn function_defs(&self) -> String {
		self.program.structs.iter()
						 	.map(|strct|
						 		strct.steps.iter()
										   .map(|step| self.step_call_function_as_c(strct, step))
						 	)
						 	.flatten()
						 	.collect::<Vec<String>>()
						 	.join("\n")
	}

	fn run_schedule(&self) -> String {
		let schedule = self.schedule_as_c(&self.program.schedule, 1);
		let needs_initial_fp_context = !matches!(*self.program.schedule, Schedule::Fixpoint(..));

		let push = if needs_initial_fp_context {
			"host_FP.push();".to_string()
		} else {
			"".to_string()
		};

		let instance_counters;
		if self.printnthinst != 0 {
			instance_counters = self.program.structs.iter()
													.map(|s| format!("\tinst_size last_nrof_{}s = 0;", s.name))
													.collect::<Vec<String>>()
													.join("\n");
		} else {
			instance_counters = "\n".to_string();
		}

		let was_done = if self.print_unstable {
			"\tbool was_done = host_FP.done();\n"
		} else {
			""
		};
		
		formatdoc!{"
			{push}
				device_FP = host_FP.to_device();
			{instance_counters}
			{was_done}
			{schedule}
		"}
	}
}


impl CoalescedScheduleManager<'_> {
	pub fn new<'a>(program: &'a Program, struct_manager : &'a dyn StructManager, printnthinst : usize, print_unstable: bool) -> CoalescedScheduleManager<'a> {
		CoalescedScheduleManager {
			program,
			step_to_structs : program.get_step_to_structs(),
			struct_manager,
			tpb: 512,
			printnthinst,
			print_unstable
		}
	}

	fn step_call_function_as_c(&self, strct: &ADLStruct, step: &Step) -> String {
		let strct_name = &strct.name;
		
		let arg_array = self.struct_manager.kernel_arguments(strct, step)
										   .join(&format!(",\n\t\t"));
		let kernel_name = self.struct_manager.kernel_name(strct, step);

		formatdoc!("
			void launch_{kernel_name}() {{
				inst_size nrof_instances = host_{strct_name}.nrof_instances();
				void* {kernel_name}_args[] = {{
					{arg_array}
				}};
				auto dims = ADL::get_launch_dims(nrof_instances, (void*){kernel_name});

				CHECK(
					cudaLaunchCooperativeKernel(
						(void*){kernel_name},
						std::get<0>(dims),
						std::get<1>(dims),
						{kernel_name}_args
					)
				);
				CHECK(cudaDeviceSynchronize());
			}}
		")
	}

	fn step_call_as_c(&self, strct: &ADLStruct, step: &Step, indent_lvl : usize) -> String {
		let indent = "\t".repeat(indent_lvl);
		let kernel_name = self.struct_manager.kernel_name(strct, step);
		let strct_name = &strct.name;

		let printnthinst = self.printnthinst;
		let printer;
		if printnthinst != 0 {
			printer = formatdoc!{"
				{indent}if (host_{strct_name}.nrof_instances() - last_nrof_{strct_name}s >= {printnthinst}) {{
				{indent}	last_nrof_{strct_name}s = host_{strct_name}.nrof_instances();
				{indent}	fprintf(stderr, \"Created %u {strct_name}s\\n\", last_nrof_{strct_name}s);
				{indent}}}
			"};
		} else {
			printer = "".to_string();
		}

		let prev_stability = if self.print_unstable {
			formatdoc!("
				{indent}was_done = host_FP.done();
				{indent}host_FP.reset(); // Set to done
				{indent}host_FP.copy_to(device_FP);
			")
		} else {
			"".to_string()
		};

		let print_stability = if self.print_unstable {
			formatdoc!("
				{indent}host_FP.copy_from(device_FP);
				{indent}if(!host_FP.done()){{
				{indent}	fprintf(stderr, \"{kernel_name} was unstable.\\n\");
				{indent}}}
				{indent}if(!was_done) {{
				{indent}	host_FP.set(); // set back to not done if needed
				{indent}}}
				{indent}host_FP.copy_to(device_FP);
				")
		} else {
			"".to_string()
		};


		formatdoc!("
			{prev_stability}
			{indent}launch_{kernel_name}();
			{print_stability}
			{printer}
		")
	}

	fn call_print_as_c(&self, struct_name: &String, indent_lvl : usize) -> String {
		let indent = "\t".repeat(indent_lvl);
		let tpb = self.tpb;
		let kernel_name = format!("{struct_name}_print");

		formatdoc!("
			{indent}{kernel_name}<<<(host_{struct_name}.nrof_instances() + {tpb} - 1)/{tpb}, {tpb}>>>(gm_{struct_name}, host_{struct_name}.nrof_instances());
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
					{indent}host_FP.copy_from(device_FP);
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
					{indent}host_FP.copy_to(device_FP);
				"};
			}
		}
	}
}