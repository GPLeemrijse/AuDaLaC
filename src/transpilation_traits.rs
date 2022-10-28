use std::collections::BTreeSet;
use crate::ast::*;


pub trait Transpiler {
    fn transpile(program: &Program, schedule_manager: &impl ScheduleManager) -> String;
}


pub trait ScheduleManager {
    fn add_includes(&self, set: &mut BTreeSet<String>);
    fn defines(&self) -> String;
    fn struct_typedef(&self) -> String;
    fn globals(&self) -> String;
    fn function_defs(&self) -> String;
    fn run_schedule(&self) -> String;
    fn call_reset_stability(&self) -> String;
}

pub trait StructManager {
    fn includes(&self) -> String;
    fn defines(&self) -> String;
    fn struct_typedef(&self) -> String;
    fn globals(&self) -> String;
    fn function_defs(&self) -> String;
    fn pre_main(&self) -> String;
    fn post_main(&self) -> String;
}
