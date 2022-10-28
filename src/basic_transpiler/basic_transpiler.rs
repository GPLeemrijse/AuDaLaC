use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::Program;





pub struct BasicCUDATranspiler {
}

impl Transpiler for BasicCUDATranspiler {
	fn transpile(program: &Program, schedule_manager : &impl ScheduleManager) -> String {
		let mut result = String::new();
		let mut includes : BTreeSet<String> = BTreeSet::new();
		schedule_manager.add_includes(&mut includes);

		for i in includes {
			result.push_str(&format!("#include {}\n", i));
		}

		result.push_str(&schedule_manager.defines());
		result
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