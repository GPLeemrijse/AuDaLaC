use std::collections::HashMap;
use std::collections::BTreeSet;
use crate::transpilation_traits::*;
use crate::ast::*;


pub struct CoalescedScheduleManager<'a> {
	program : &'a Program,
	step_to_structs : HashMap<String, Vec<String>>,
}

impl ScheduleManager for CoalescedScheduleManager<'_> {
	fn add_includes(&self, _: &mut BTreeSet<std::string::String>) { todo!() }
	fn defines(&self) -> std::string::String { todo!() }
	fn struct_typedef(&self) -> std::string::String { todo!() }
	fn globals(&self) -> std::string::String { todo!() }
	fn function_defs(&self) -> std::string::String { todo!() }
	fn run_schedule(&self) -> std::string::String { todo!() }
	fn call_set_dirty(&self) -> std::string::String { todo!() }
}


impl CoalescedScheduleManager<'_> {
	pub fn new(program: &Program) -> CoalescedScheduleManager {
		CoalescedScheduleManager {
			program,
			step_to_structs : HashMap::new()
		}
	}
}