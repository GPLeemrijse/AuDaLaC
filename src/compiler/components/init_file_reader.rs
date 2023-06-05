use crate::compiler::CompileComponent;
use indoc::formatdoc;
use std::collections::BTreeSet;

pub struct InitFileReader {}

impl CompileComponent for InitFileReader {
    fn add_includes(&self, set: &mut BTreeSet<&str>) {
        set.insert("<stdio.h>");
        set.insert("<vector>");
    }

    fn defines(&self) -> Option<String> {
        None
    }
    fn typedefs(&self) -> Option<String> {
        None
    }
    fn globals(&self) -> Option<String> {
        None
    }
    fn functions(&self) -> Option<String> {
        None
    }
    fn kernels(&self) -> Option<String> {
        None
    }

    fn pre_main(&self) -> Option<String> {
        Some(formatdoc! {"
			\tif (argc != 2) {{
			\t\tprintf(\"Supply a .init file.\\n\");
			\t\texit(1);
			\t}}
			
			\tstd::vector<InitFile::StructInfo> structs = InitFile::parse(argv[1]);
		"})
    }

    fn main(&self) -> Option<String> {
        None
    }
    fn post_main(&self) -> Option<String> {
        None
    }
}
