use std::collections::BTreeSet;

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

pub trait FPStrategy {
    fn global_decl(&self) -> String;
    fn top_of_kernel_decl(&self) -> String;
    fn post_kernel(&self) -> String;
    fn pre_iteration(&self, lvl: usize) -> String;
    fn post_iteration(&self, lvl: usize) -> String;
    fn is_stable(&self, lvl: usize) -> String;

    fn initialise(&self) -> String;
    fn requires_intra_fixpoint_sync(&self) -> bool;
}
