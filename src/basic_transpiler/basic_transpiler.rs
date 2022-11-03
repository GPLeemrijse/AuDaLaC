use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::Program;
use indoc::formatdoc;



pub struct BasicCUDATranspiler {
}

impl Transpiler for BasicCUDATranspiler {
	fn transpile(program: &Program, schedule_manager : &impl ScheduleManager) -> String {
		let mut includes = String::new();
		let mut defines = String::new();
		let mut typedefs = String::new();
		let mut globals = String::new();
		let mut functs = String::new();

		let mut includes_set : BTreeSet<String> = BTreeSet::new();
		schedule_manager.add_includes(&mut includes_set);

		for i in includes_set {
			includes.push_str(&format!("#include {}\n", i));
		}

		defines.push_str(&schedule_manager.defines());
		typedefs.push_str(&schedule_manager.struct_typedef());
		globals.push_str(&schedule_manager.globals());
		functs.push_str(&schedule_manager.function_defs());

		let schedule = schedule_manager.run_schedule();
		println!("{}", schedule);
		formatdoc! {"
			{includes}
			{defines}
			{typedefs}
			{globals}
			{functs}

			int main() {{
			    {schedule}
			}}
		"}
	}
}

impl BasicCUDATranspiler {
	
}


static SET_PARAM: &str = r#"#define SET_PARAM(T, P, V) ({T read_val = P; if (read_val != V) {P = V; clear_stability_stack();}})"#;

static STRUCT_MANAGER : &str = r#"typedef struct StructManager {
	void* structs;
	unsigned int struct_size;
	unsigned int nrof_active_structs;
	unsigned int nrof_active_structs_before_launch;
} StructManager;"#;

static FP_MANAGER : &str = r#"typedef struct FixpointManager {
	bool stack[FP_DEPTH];
	unsigned int current_level;
} FixpointManager;"#;