use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use indoc::formatdoc;


pub fn transpile(schedule_manager : &dyn ScheduleManager, struct_manager : &dyn StructManager) -> String {
	let mut includes = String::new();
	let mut defines = String::new();
	let mut typedefs = String::new();
	let mut globals = String::new();
	let mut functs = String::new();
	let mut pre_main = String::new();
	let mut post_main = String::new();
	let kernels = struct_manager.kernels();

	let mut includes_set : BTreeSet<&str> = BTreeSet::from([
		// INIT FILE
		"<stdio.h>",
		"<vector>"
	]);
	
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
		{defines}

		{includes}

		{typedefs}

		{globals}

		{kernels}

		{functs}

		int main(int argc, char **argv) {{
			{pre_main}

			size_t printf_size;
			cudaDeviceGetLimit(&printf_size, cudaLimitPrintfFifoSize);
			cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 2 * printf_size);

			{schedule}

			{post_main}
		}}
	"}
}

pub fn transpile2(components : Vec<&dyn CompileComponent>) -> String {
	// Program sections
	let (mut includes, mut defines,
		 mut typedefs, mut globals,
		 mut functs, mut pre_main,
		 mut post_main, mut main,
		 mut kernels) = (String::new(), String::new(),
						 String::new(), String::new(),
						 String::new(), String::new(),
						 String::new(), String::new(),
						 String::new());

	let mut includes_set : BTreeSet<&str> = BTreeSet::new();

	let mut sec2func_map : Vec<(&mut String, Box<dyn Fn(&dyn CompileComponent) -> Option<String>>)> = vec![
		(&mut defines,   Box::new(|c| c.defines())),
		(&mut typedefs,  Box::new(|c| c.typedefs())),
		(&mut globals,   Box::new(|c| c.globals())),
		(&mut functs,    Box::new(|c| c.functions())),
		(&mut pre_main,  Box::new(|c| c.pre_main())),
		(&mut post_main, Box::new(|c| c.post_main())),
		(&mut main,      Box::new(|c| c.main())),
		(&mut kernels,   Box::new(|c| c.kernels()))
	];

	for c in components {
		for (sec, f) in &mut sec2func_map {
			if let Some(src) = f(c) {
				sec.push_str(&src);
			}
		}

		c.add_includes(&mut includes_set);
	}

	for i in includes_set {
		includes.push_str(&format!("#include {}\n", i));
	}

	formatdoc! {"
		{defines}

		{includes}

		{typedefs}

		{globals}

		{kernels}

		{functs}

		int main(int argc, char **argv) {{
		{pre_main}

		{main}

		{post_main}
		}}
	"}
}