use std::collections::BTreeSet;

pub trait Transpiler {
    fn transpile(schedule_manager: &impl ScheduleManager, struct_manager: &impl StructManager) -> String;
}


pub trait ScheduleManager {
    fn add_includes(&self, set: &mut BTreeSet<String>);
    fn defines(&self) -> String;
    fn struct_typedef(&self) -> String;
    fn globals(&self) -> String;
    fn function_defs(&self) -> String;
    fn run_schedule(&self) -> String;
    fn call_set_dirty(&self) -> String;
}

pub trait StructManager {
    fn add_includes(&self, set: &mut BTreeSet<String>);
    fn defines(&self) -> String;
    fn struct_typedef(&self) -> String;
    fn globals(&self) -> String;
    fn function_defs(&self) -> String;
    fn kernels(&self) -> String;
    fn pre_main(&self) -> String;
    fn post_main(&self) -> String;
}
