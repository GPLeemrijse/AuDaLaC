use std::collections::BTreeSet;
use crate::ast::{ADLStruct, Step};

pub trait Transpiler {
    fn transpile(schedule_manager: &dyn ScheduleManager, struct_manager: &dyn StructManager) -> String;
}


pub trait ScheduleManager {
    fn add_includes(&self, set: &mut BTreeSet<&str>);
    fn defines(&self) -> String;
    fn struct_typedef(&self) -> String;
    fn globals(&self) -> String;
    fn function_defs(&self) -> String;
    fn run_schedule(&self) -> String;
}

pub trait StructManager {
    fn add_includes(&self, set: &mut BTreeSet<&str>);
    fn defines(&self) -> String;
    fn struct_typedef(&self) -> String;
    fn globals(&self) -> String;
    fn function_defs(&self) -> String;
    fn kernels(&self) -> String;
    fn pre_main(&self) -> String;
    fn post_main(&self) -> String;

    fn kernel_parameters(&self, strct: &ADLStruct, step: &Step) -> Vec<String>;
    fn kernel_arguments(&self, strct: &ADLStruct, step: &Step) -> Vec<String>;
    fn kernel_name(&self, strct: &ADLStruct, step: &Step) -> String;
}
