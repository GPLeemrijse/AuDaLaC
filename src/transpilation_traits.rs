use std::collections::BTreeSet;
use crate::ast::{ADLStruct, Step};


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


pub trait CompileComponent {
    fn add_includes(&self, set: &mut BTreeSet<&str>);
    fn defines(&self) -> Option<String>;
    fn typedefs(&self) -> Option<String>;
    fn globals(&self) -> Option<String>;
    fn functions(&self) -> Option<String>;
    fn kernels(&self) -> Option<String>;
    fn pre_main(&self) -> Option<String>;
    fn main(&self) -> Option<String>;
    fn post_main(&self) -> Option<String>;
}