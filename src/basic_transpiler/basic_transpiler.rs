use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;



pub struct BasicCUDATranspiler {
}

impl Transpiler for BasicCUDATranspiler {
	fn transpile(schedule_manager : &impl ScheduleManager, struct_manager : &impl StructManager) -> String {
		let mut includes = String::new();
		let mut defines = String::new();
		let mut typedefs = String::new();
		let mut globals = String::new();
		let mut functs = String::new();
		let mut pre_main = String::new();
		let mut post_main = String::new();
		let kernels = struct_manager.kernels();

		let mut includes_set : BTreeSet<String> = BTreeSet::new();
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
			#include <sys/time.h>
			{includes}

			{defines}

			{typedefs}

			{globals}

			{functs}

			{kernels}

			int main() {{
				struct timeval t1, t2;

				gettimeofday(&t1, 0);

				{pre_main}

			    {schedule}

				{post_main}

				gettimeofday(&t2, 0);
				double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
				fprintf(stderr, \"%3.1f ms\\n\", time);
			}}
		"}
	}
}

impl BasicCUDATranspiler {
	
}
