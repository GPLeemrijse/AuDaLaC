use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;


pub struct BasicScheduleManager<'a> {
	program : &'a Program,
}

impl ScheduleManager for BasicScheduleManager<'_> {
	fn add_includes(&self, set: &mut BTreeSet<String>) {
		set.insert("<stdio.h>".to_string());
	}

	fn defines(&self) -> std::string::String {
		format!("#define FP_DEPTH {}\n", self.program.schedule.fixpoint_depth())
	}

	fn struct_typedef(&self) -> std::string::String { todo!() }
	fn globals(&self) -> std::string::String { todo!() }
	fn function_defs(&self) -> std::string::String { todo!() }
	fn run_schedule(&self) -> std::string::String { todo!() }
	fn call_reset_stability(&self) -> std::string::String { todo!() }
}


impl BasicScheduleManager<'_> {
	pub fn new(program: &Program) -> BasicScheduleManager {
		BasicScheduleManager {
			program
		}
	}
}