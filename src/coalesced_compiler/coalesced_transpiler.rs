use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;



pub struct CoalescedCUDATranspiler {
}

impl Transpiler for CoalescedCUDATranspiler {
	fn transpile(schedule_manager : &dyn ScheduleManager, struct_manager : &dyn StructManager) -> String {
		let mut includes = String::new();
		let mut defines = String::new();
		let mut typedefs = String::new();
		let mut globals = String::new();
		let mut functs = String::new();
		let mut pre_main = String::new();
		let mut post_main = String::new();
		let kernels = struct_manager.kernels();

		let mut includes_set : BTreeSet<String> = BTreeSet::new();
		
		// INIT FILE
		includes_set.insert("<stdio.h>".to_string());
		includes_set.insert("<vector>".to_string());
		pre_main.push_str(&formatdoc!{"
		if (argc != 2) {{
				printf(\"Supply a .init file.\\n\");
				exit(1);
			}}
			
			std::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
		"});

		
		schedule_manager.add_includes(&mut includes_set);
		struct_manager.add_includes(&mut includes_set);

		for i in includes_set {
			includes.push_str(&format!("#include {}\n", i));
		}

		defines.push_str(&schedule_manager.defines());
		defines.push_str(&struct_manager.defines());
		typedefs.push_str(&schedule_manager.struct_typedef());
		typedefs.push_str(&struct_manager.struct_typedef());
		globals.push_str(&schedule_manager.globals());
		globals.push_str(&struct_manager.globals());
		functs.push_str(&schedule_manager.function_defs());
		functs.push_str(&struct_manager.function_defs());
		pre_main.push_str(&struct_manager.pre_main());
		post_main.push_str(&struct_manager.post_main());


		let schedule = schedule_manager.run_schedule();
		formatdoc! {"
			{includes}

			{defines}

			{typedefs}

			{globals}

			{functs}

			{kernels}

			int main(int argc, char **argv) {{
				{pre_main}

				{schedule}

				{post_main}
			}}
		"}
	}
}